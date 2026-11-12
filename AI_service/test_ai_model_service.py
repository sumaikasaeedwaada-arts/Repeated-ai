import os
import sqlite3
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from google.genai import errors as genai_errors

from AI_service import database
from AI_service.errors import (
    AIProviderAuthenticationError,
    AIProviderError,
    AIProviderResponseError,
    AIProviderTimeoutError,
)
from AI_service.services import ai_model_service


class AIModelServiceTests(unittest.TestCase):
    def _initialize_cache_database(self, user_count=2):
        database.initialize_database()
        with database.connect() as connection:
            for user_id in range(1, user_count + 1):
                connection.execute(
                    "INSERT INTO users(id, email, password_hash, created_at) "
                    "VALUES (?, ?, 'test-hash', '2026-01-01T00:00:00+00:00')",
                    (user_id, f"user{user_id}@example.com")
                )

    def test_existing_database_migration_preserves_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, "existing.sqlite")
            connection = sqlite3.connect(db_path)
            try:
                connection.executescript(
                    """
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                        password_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE conversations (
                        id TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id),
                        title TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL
                            REFERENCES conversations(id),
                        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        citations_json TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL
                    );
                    INSERT INTO users(email, password_hash, created_at)
                    VALUES ('owner@example.com', 'test-hash', '2026-01-01');
                    INSERT INTO conversations
                        (id, user_id, title, created_at, updated_at)
                    VALUES ('conversation', 1, 'Existing', '2026-01-01', '2026-01-01');
                    INSERT INTO messages(conversation_id, role, content, created_at)
                    VALUES ('conversation', 'user', 'Existing message', '2026-01-01');
                    """
                )
            finally:
                connection.close()

            with patch.dict(os.environ, {"DATABASE_PATH": db_path}):
                database.initialize_database()
                with database.connect() as connection:
                    message = connection.execute(
                        "SELECT content, request_fingerprint FROM messages"
                    ).fetchone()

        self.assertEqual(message["content"], "Existing message")
        self.assertIsNone(message["request_fingerprint"])

    def test_invalid_key_is_sanitized_and_logged_without_provider_details(self):
        for status, provider_message in (
            (401, "secret-provider-response"),
            (400, "API key not valid: secret-provider-response")
        ):
            api_error = genai_errors.ClientError(
                status,
                {"message": provider_message}
            )
            with (
                self.subTest(status=status),
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                patch.object(
                    ai_model_service.client.models,
                    "generate_content",
                    side_effect=api_error
                ),
                self.assertLogs(
                    "AI_service.services.ai_model_service",
                    level="WARNING"
                ) as captured
            ):
                with self.assertRaises(AIProviderAuthenticationError):
                    ai_model_service.generate_answer(
                        "private customer question"
                    )

            logged = "\n".join(
                record.getMessage() for record in captured.records
            )
            self.assertNotIn("secret-provider-response", logged)
            self.assertNotIn("private customer question", logged)
            self.assertEqual(
                captured.records[0].failure_kind,
                "authentication"
            )
            self.assertEqual(captured.records[0].provider_status, status)

    def test_timeout_and_connection_failures_are_classified(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch.object(
                ai_model_service.client.models,
                "generate_content",
                side_effect=httpx.ReadTimeout("sensitive timeout detail")
            ):
                with self.assertRaises(AIProviderTimeoutError):
                    ai_model_service.generate_answer("question")
            with patch.object(
                ai_model_service.client.models,
                "generate_content",
                side_effect=httpx.ConnectError("sensitive host detail")
            ):
                with self.assertRaises(AIProviderError) as raised:
                    ai_model_service.generate_answer("question")
        self.assertNotIsInstance(raised.exception, AIProviderTimeoutError)

    def test_empty_and_malformed_text_are_rejected(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            for response in (
                SimpleNamespace(text=" \n "),
                SimpleNamespace(text={"text": "not a string"}),
                SimpleNamespace()
            ):
                with (
                    self.subTest(response=response),
                    patch.object(
                        ai_model_service.client.models,
                        "generate_content",
                        return_value=response
                    ),
                    self.assertRaises(AIProviderResponseError)
                ):
                    ai_model_service.generate_answer("question")

    def test_valid_response_is_returned_as_trimmed_text(self):
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.object(
                ai_model_service.client.models,
                "generate_content",
                return_value=SimpleNamespace(text="  Safe answer. \n")
            )
        ):
            result = ai_model_service.generate_answer("question")

        self.assertEqual(result, "Safe answer.")

    def test_response_cache_is_user_scoped_and_prompt_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, "cache.sqlite")
            with patch.dict(os.environ, {
                "DATABASE_PATH": db_path,
                "GEMINI_API_KEY": "test-key"
            }):
                self._initialize_cache_database()
                with patch.object(
                    ai_model_service.client.models,
                    "generate_content",
                    return_value=SimpleNamespace(text="Scoped answer.")
                ) as generate:
                    first = ai_model_service.generate_answer(
                        "same prompt",
                        cache_user_id=1
                    )
                    repeated = ai_model_service.generate_answer(
                        "same prompt",
                        cache_user_id=1
                    )
                    other_user = ai_model_service.generate_answer(
                        "same prompt",
                        cache_user_id=2
                    )
                    other_prompt = ai_model_service.generate_answer(
                        "different prompt",
                        cache_user_id=1
                    )

        self.assertEqual(
            [first, repeated, other_user, other_prompt],
            ["Scoped answer."] * 4
        )
        self.assertEqual(generate.call_count, 3)

    def test_concurrent_identical_requests_share_one_provider_call(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, "cache.sqlite")
            with patch.dict(os.environ, {
                "DATABASE_PATH": db_path,
                "GEMINI_API_KEY": "test-key"
            }):
                self._initialize_cache_database(user_count=1)
                barrier = threading.Barrier(2)

                def generate(**request):
                    time.sleep(0.15)
                    self.assertEqual(request["contents"], "identical prompt")
                    return SimpleNamespace(text="One provider response.")

                def request_answer():
                    barrier.wait()
                    return ai_model_service.generate_answer(
                        "identical prompt",
                        cache_user_id=1
                    )

                with patch.object(
                    ai_model_service.client.models,
                    "generate_content",
                    side_effect=generate
                ) as provider_call:
                    results = []
                    threads = [
                        threading.Thread(
                            target=lambda: results.append(request_answer())
                        )
                        for _ in range(2)
                    ]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join(timeout=10)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(results, ["One provider response."] * 2)
        self.assertEqual(provider_call.call_count, 1)

    def test_embedding_cache_is_scoped_by_user_and_task_type(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, "cache.sqlite")
            with patch.dict(os.environ, {
                "DATABASE_PATH": db_path,
                "GEMINI_API_KEY": "test-key"
            }):
                self._initialize_cache_database()
                embedding = SimpleNamespace(
                    embeddings=[SimpleNamespace(values=[0.25, 0.75])]
                )
                with patch.object(
                    ai_model_service.client.models,
                    "embed_content",
                    return_value=embedding
                ) as embed:
                    first = ai_model_service.create_embedding(
                        "private text",
                        "RETRIEVAL_DOCUMENT",
                        cache_user_id=1
                    )
                    repeated = ai_model_service.create_embedding(
                        "private text",
                        "RETRIEVAL_DOCUMENT",
                        cache_user_id=1
                    )
                    other_user = ai_model_service.create_embedding(
                        "private text",
                        "RETRIEVAL_DOCUMENT",
                        cache_user_id=2
                    )
                    other_task = ai_model_service.create_embedding(
                        "private text",
                        "RETRIEVAL_QUERY",
                        cache_user_id=1
                    )

        self.assertEqual(first, [0.25, 0.75])
        self.assertEqual(repeated, first)
        self.assertEqual(other_user, first)
        self.assertEqual(other_task, first)
        self.assertEqual(embed.call_count, 3)


if __name__ == "__main__":
    unittest.main()
