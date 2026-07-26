import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.database import initialize_database
from app.repositories.document_repository import ChunkInput, add_chunks, create_document
from app.services.rag.service import (
    INSUFFICIENT_CONTEXT_TOKEN,
    _validation_issues,
    answer_question,
)
from app.services.retrieval import QuestionScope, RetrievedChunk


class AnswerValidationTests(unittest.TestCase):
    def test_condition_answer_must_preserve_the_source_action(self) -> None:
        source = RetrievedChunk(
            chunk_id=1,
            document_id=1,
            file_name="GDPR-2016-679-EN.pdf",
            content=(
                "When a personal data breach is likely to result in a high risk, "
                "the controller shall, without undue delay, communicate the personal "
                "data breach to the data subject."
            ),
            page_number=1,
            section="Article 34",
            article="Article 34",
            paragraph="1",
            point=None,
            subpoint=None,
            score=0.9,
        )

        issues = _validation_issues(
            "When the breach is likely to result in a high risk, the controller must "
            "without delay discuss the breach with the data subject.",
            "When must a personal data breach be communicated to the data subject?",
            QuestionScope.FOCUSED,
            [source],
        )

        self.assertIn("missing legal action: communicate", issues)

    def test_obligation_answer_must_preserve_the_source_action(self) -> None:
        source = RetrievedChunk(
            chunk_id=2,
            document_id=1,
            file_name="EU-AI-Act-2024-1689-EN.pdf",
            content="The placing on the market of this AI practice shall be prohibited.",
            page_number=1,
            section="Article 5",
            article="Article 5",
            paragraph="1",
            point="g",
            subpoint=None,
            score=0.9,
        )

        issues = _validation_issues(
            "This AI practice is regulated under the Regulation.",
            "What practices are prohibited?",
            QuestionScope.FOCUSED,
            [source],
        )

        self.assertIn("missing legal action: prohibit", issues)

    def test_focused_answer_rejects_an_unrequested_article_reference(self) -> None:
        source = RetrievedChunk(
            chunk_id=3,
            document_id=1,
            file_name="EU-AI-Act-2024-1689-EN.pdf",
            content="The use of biometric categorisation systems for this purpose shall be prohibited.",
            page_number=1,
            section="Article 5",
            article="Article 5",
            paragraph="1",
            point="g",
            subpoint=None,
            score=0.9,
        )

        issues = _validation_issues(
            "Article 5(1)(g) prohibits this use of biometric categorisation systems.",
            "What biometric categorisation practices are prohibited?",
            QuestionScope.FOCUSED,
            [source],
        )

        self.assertIn("unrequested legal reference", issues)

    def test_validation_retry_uses_the_source_when_it_claims_insufficient_context(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "validation-retry.db"
            initialize_database(database_path)
            document = create_document(
                original_name="EU-AI-Act-2024-1689-EN.pdf",
                stored_name="ai-act.pdf",
                file_type="pdf",
                file_size=1,
                content_hash="validation-retry",
                database_path=database_path,
            )
            add_chunks(
                document["id"],
                [
                    ChunkInput(
                        content="The following AI practices shall be prohibited:",
                        embedding=[1.0, 0.0],
                        article="Article 5",
                        paragraph="1",
                    ),
                    ChunkInput(
                        content="(g) The use of biometric categorisation systems for this purpose shall be prohibited.",
                        embedding=[1.0, 0.0],
                        article="Article 5",
                        paragraph="1",
                        point="g",
                    )
                ],
                database_path,
            )

            responses = iter(
                [
                    "Article 5(1)(g) prohibits this biometric categorisation practice.",
                    INSUFFICIENT_CONTEXT_TOKEN,
                ]
            )
            result = answer_question(
                "What biometric categorisation practices are prohibited?",
                database_path=database_path,
                embedding_function=lambda _: [[1.0, 0.0]],
                chat_function=lambda _: next(responses),
            )

        self.assertIn("prohibited", result.answer.casefold())
        self.assertNotEqual(result.answer, INSUFFICIENT_CONTEXT_TOKEN)

    def test_legal_answer_uses_the_source_when_the_initial_generation_claims_insufficient_context(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "initial-insufficient.db"
            initialize_database(database_path)
            document = create_document(
                original_name="EU-AI-Act-2024-1689-EN.pdf",
                stored_name="ai-act.pdf",
                file_type="pdf",
                file_size=1,
                content_hash="initial-insufficient",
                database_path=database_path,
            )
            add_chunks(
                document["id"],
                [
                    ChunkInput(
                        content="(g) The use of biometric categorisation systems for this purpose shall be prohibited.",
                        embedding=[1.0, 0.0],
                        article="Article 5",
                        paragraph="1",
                        point="g",
                    )
                ],
                database_path,
            )

            result = answer_question(
                "What biometric categorisation practices are prohibited?",
                database_path=database_path,
                embedding_function=lambda _: [[1.0, 0.0]],
                chat_function=lambda _: INSUFFICIENT_CONTEXT_TOKEN,
            )

        self.assertIn("prohibited", result.answer.casefold())
        self.assertNotEqual(result.answer, INSUFFICIENT_CONTEXT_TOKEN)


if __name__ == "__main__":
    unittest.main()
