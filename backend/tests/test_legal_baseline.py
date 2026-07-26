import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from app.database import initialize_database
from app.services.document_processing.readers import PageText
from app.services.document_processing.legal_parser import _create_legal_blocks, is_legal_document
from app.services.rag.service import _question_scope
from app.services.retrieval import (
    QuestionScope,
    RetrievedChunk,
    _focused_clause_indices,
    _paragraph_anchor_index,
    retrieve_relevant_chunks,
)
from app.repositories.document_repository import ChunkInput, add_chunks, create_document


def legal_source(
    *,
    document_id: int,
    file_name: str,
    article: str,
    paragraph: str | None = None,
    point: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=document_id,
        document_id=document_id,
        file_name=file_name,
        content="A legal rule supported by the selected provision.",
        page_number=1,
        section=article,
        article=article,
        paragraph=paragraph,
        point=point,
        subpoint=None,
        score=0.9,
    )


class LegalBaselineTests(unittest.TestCase):
    def test_legal_parser_detects_article_structure(self) -> None:
        pages = [
            PageText(
                page=1,
                text="""CHAPTER I
Article 1
Subject matter
(1) This Regulation lays down rules.
(a) First condition.
(b) Second condition.
Article 2
Scope
(1) This Regulation applies.
Article 3
Definitions
(1) For the purposes of this Regulation.""",
            )
        ]

        self.assertTrue(is_legal_document(pages))
        blocks = _create_legal_blocks(pages)

        self.assertTrue(
            any(
                block.article == "Article 1"
                and block.paragraph == "1"
                and block.point == "a"
                for block in blocks
            )
        )
        self.assertTrue(
            any(block.article == "Article 3" and block.paragraph == "1" for block in blocks)
        )

    def test_definition_question_keeps_definition_scope(self) -> None:
        source = legal_source(
            document_id=1,
            file_name="EU-AI-Act-2024-1689-EN.pdf",
            article="Article 3",
            paragraph="1",
        )

        self.assertEqual(
            _question_scope("What is an AI system?", [source]),
            QuestionScope.DEFINITION,
        )

    def test_multi_document_question_uses_cross_document_scope(self) -> None:
        sources = [
            legal_source(
                document_id=1,
                file_name="EU-AI-Act-2024-1689-EN.pdf",
                article="Article 14",
            ),
            legal_source(
                document_id=2,
                file_name="GDPR-2016-679-EN.pdf",
                article="Article 22",
            ),
        ]

        self.assertEqual(
            _question_scope(
                "How do the GDPR and the AI Act address automated decision-making?",
                sources,
            ),
            QuestionScope.CROSS_DOCUMENT,
        )

    def test_point_level_legal_question_uses_focused_scope(self) -> None:
        source = legal_source(
            document_id=1,
            file_name="EU-AI-Act-2024-1689-EN.pdf",
            article="Article 5",
            paragraph="1",
            point="g",
        )

        self.assertEqual(
            _question_scope("What practice is prohibited?", [source]),
            QuestionScope.FOCUSED,
        )

    def test_content_question_selects_a_parent_clause_and_its_child_points(self) -> None:
        chunks = [
            {
                "document_id": 1,
                "article": "Article 33",
                "paragraph": "1",
                "point": None,
                "subpoint": None,
                "content": "The controller shall notify the supervisory authority within 72 hours.",
            },
            {
                "document_id": 1,
                "article": "Article 33",
                "paragraph": "3",
                "point": None,
                "subpoint": None,
                "content": "The notification shall at least contain the following information:",
            },
            {
                "document_id": 1,
                "article": "Article 33",
                "paragraph": "3",
                "point": "a",
                "subpoint": None,
                "content": "(a) describe the nature of the personal data breach;",
            },
            {
                "document_id": 1,
                "article": "Article 33",
                "paragraph": "3",
                "point": "b",
                "subpoint": None,
                "content": "(b) provide contact details for further information;",
            },
        ]

        anchor = _paragraph_anchor_index(
            "What information must a personal data breach notification contain?",
            chunks,
            article_anchor_index=0,
            ranking_scores=np.asarray([0.9, 0.8, 0.75, 0.7], dtype=np.float32),
        )

        self.assertEqual(anchor, 1)
        self.assertEqual(_focused_clause_indices(chunks, anchor), [1, 2, 3])

    def test_point_expansion_keeps_its_parent_rule(self) -> None:
        chunks = [
            {
                "document_id": 1,
                "article": "Article 5",
                "paragraph": "1",
                "point": None,
                "subpoint": None,
                "content": "The following AI practices shall be prohibited:",
            },
            {
                "document_id": 1,
                "article": "Article 5",
                "paragraph": "1",
                "point": "g",
                "subpoint": None,
                "content": "(g) biometric categorisation to infer sensitive characteristics;",
            },
        ]

        self.assertEqual(_focused_clause_indices(chunks, 1), [0, 1])

    def test_timing_question_prefers_the_clause_with_its_triggering_condition(self) -> None:
        chunks = [
            {
                "document_id": 1,
                "article": "Article 34",
                "paragraph": "1",
                "point": None,
                "subpoint": None,
                "content": "When a personal data breach is likely to result in a high risk, the controller shall communicate it without undue delay.",
            },
            {
                "document_id": 1,
                "article": "Article 34",
                "paragraph": "2",
                "point": None,
                "subpoint": None,
                "content": "The communication shall describe the nature of the personal data breach.",
            },
        ]

        anchor = _paragraph_anchor_index(
            "When must a personal data breach be communicated to the data subject?",
            chunks,
            article_anchor_index=1,
            ranking_scores=np.asarray([0.82, 0.90], dtype=np.float32),
        )

        self.assertEqual(anchor, 0)

    def test_cross_document_retrieval_selects_a_source_from_each_regulation(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "baseline.db"
            initialize_database(database_path)
            ai_act = create_document(
                original_name="EU-AI-Act-2024-1689-EN.pdf",
                stored_name="ai-act.pdf",
                file_type="pdf",
                file_size=1,
                content_hash="ai-act",
                database_path=database_path,
            )
            gdpr = create_document(
                original_name="GDPR-2016-679-EN.pdf",
                stored_name="gdpr.pdf",
                file_type="pdf",
                file_size=1,
                content_hash="gdpr",
                database_path=database_path,
            )
            add_chunks(
                ai_act["id"],
                [
                    ChunkInput(
                        content="Article 14 Human oversight for high-risk AI systems.",
                        embedding=[1.0, 0.0],
                        article="Article 14",
                        paragraph="1",
                    )
                ],
                database_path,
            )
            add_chunks(
                gdpr["id"],
                [
                    ChunkInput(
                        content="Article 22 Automated individual decision-making safeguards.",
                        embedding=[1.0, 0.0],
                        article="Article 22",
                        paragraph="1",
                    )
                ],
                database_path,
            )

            sources = retrieve_relevant_chunks(
                "How do the GDPR and the AI Act address automated decision-making?",
                database_path=database_path,
                embedding_function=lambda _: [[1.0, 0.0]],
            )

        self.assertEqual({source.document_id for source in sources}, {1, 2})
        self.assertEqual(
            {source.article for source in sources},
            {"Article 14", "Article 22"},
        )

    def test_content_retrieval_prefers_the_direct_requirement(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "content-retrieval.db"
            initialize_database(database_path)
            gdpr = create_document(
                original_name="GDPR-2016-679-EN.pdf",
                stored_name="gdpr.pdf",
                file_type="pdf",
                file_size=1,
                content_hash="gdpr-content",
                database_path=database_path,
            )
            add_chunks(
                gdpr["id"],
                [
                    ChunkInput(
                        content="The notification shall at least contain the following information about the personal data breach.",
                        embedding=[0.8, 0.6],
                        article="Article 33",
                        paragraph="3",
                    ),
                    ChunkInput(
                        content="The communication shall contain information referred to in Article 33 for the personal data breach.",
                        embedding=[1.0, 0.0],
                        article="Article 34",
                        paragraph="2",
                    ),
                ],
                database_path,
            )

            sources = retrieve_relevant_chunks(
                "What information must a personal data breach notification contain?",
                database_path=database_path,
                embedding_function=lambda _: [[1.0, 0.0]],
            )

        self.assertEqual(sources[0].article, "Article 33")

    def test_legal_parent_and_child_terms_can_select_the_same_rule(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "hierarchy-context.db"
            initialize_database(database_path)
            regulation = create_document(
                original_name="Regulation.pdf",
                stored_name="regulation.pdf",
                file_type="pdf",
                file_size=1,
                content_hash="hierarchy-context",
                database_path=database_path,
            )
            add_chunks(
                regulation["id"],
                [
                    ChunkInput(
                        content="The following AI practices shall be prohibited:",
                        embedding=[0.0, 1.0],
                        article="Article 5",
                        paragraph="1",
                    ),
                    ChunkInput(
                        content="(g) Biometric categorisation systems may not infer sensitive characteristics.",
                        embedding=[0.0, 1.0],
                        article="Article 5",
                        paragraph="1",
                        point="g",
                    ),
                    ChunkInput(
                        content="General provisions apply to the Regulation.",
                        embedding=[0.2, 0.98],
                        article="Article 1",
                        paragraph="1",
                    ),
                ],
                database_path,
            )

            sources = retrieve_relevant_chunks(
                "What biometric categorisation practices are prohibited?",
                database_path=database_path,
                embedding_function=lambda _: [[1.0, 0.0]],
            )

        self.assertEqual({source.article for source in sources}, {"Article 5"})
        self.assertEqual({source.point for source in sources}, {None, "g"})


if __name__ == "__main__":
    unittest.main()
