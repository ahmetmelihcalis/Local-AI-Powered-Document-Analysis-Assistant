import unittest

import numpy as np

from app.services.retrieval.questions import QuestionType, build_retrieval_plan
from app.services.retrieval import _comparison_clause_indices
from app.services.retrieval.expansion import governing_condition_indices


class RetrievalPlanningTests(unittest.TestCase):
    def test_list_requires_complete_list_context(self) -> None:
        plan = build_retrieval_plan("List the notification requirements.")

        self.assertEqual(plan.question_type, QuestionType.LIST)
        self.assertTrue(plan.requires_complete_list)

    def test_procedure_and_summary_expand_their_section(self) -> None:
        for question in (
            "How do I install the application?",
            "Summarise the installation section.",
        ):
            with self.subTest(question=question):
                self.assertTrue(build_retrieval_plan(question).prefers_section_context)

    def test_comparison_requires_balanced_sources(self) -> None:
        plan = build_retrieval_plan("How do the GDPR and the AI Act differ?")

        self.assertTrue(plan.requires_balanced_sources)

    def test_broad_condition_questions_require_the_governing_context(self) -> None:
        for question in (
            "When may a data subject be subject to automated decision-making?",
            "Can an AI system make decisions about people?",
            "an ai make decisions about people",
            "When does a company need to report a serious incident?",
        ):
            with self.subTest(question=question):
                self.assertTrue(
                    build_retrieval_plan(question).requires_governing_context
                )

    def test_project_questions_prefer_general_documents(self) -> None:
        plan = build_retrieval_plan("What are the project prerequisites?")

        self.assertTrue(plan.prefers_general_documents)

    def test_governing_condition_context_includes_related_exceptions(self) -> None:
        chunks = [
            {
                "document_id": 1,
                "article": "Article 22",
                "paragraph": "1",
                "point": None,
                "subpoint": None,
                "content": "A person has the right not to be subject to automated decisions.",
            },
            {
                "document_id": 1,
                "article": "Article 22",
                "paragraph": "2",
                "point": None,
                "subpoint": None,
                "content": "Paragraph 1 shall not apply if the decision:",
            },
            {
                "document_id": 1,
                "article": "Article 22",
                "paragraph": "2",
                "point": "a",
                "subpoint": None,
                "content": "(a) is necessary for a contract.",
            },
            {
                "document_id": 1,
                "article": "Article 22",
                "paragraph": "3",
                "point": None,
                "subpoint": None,
                "content": "In cases referred to in paragraph 2, safeguards apply.",
            },
            {
                "document_id": 1,
                "article": "Article 22",
                "paragraph": "5",
                "point": None,
                "subpoint": None,
                "content": "The controller shall document its assessment.",
            },
        ]

        selected = governing_condition_indices(chunks, anchor_index=2)

        self.assertEqual(selected, [0, 1, 2, 3])

    def test_comparison_keeps_distinct_operative_articles_from_one_document(self) -> None:
        chunks = [
            {
                "document_id": 1,
                "original_name": "EU-AI-Act-2024-1689-EN.pdf",
                "article": "Article 50",
                "paragraph": "3",
                "point": None,
                "subpoint": None,
                "section": "Transparency obligations",
                "content": "Deployers of biometric categorisation systems shall inform exposed natural persons.",
            },
            {
                "document_id": 1,
                "original_name": "EU-AI-Act-2024-1689-EN.pdf",
                "article": "Article 50",
                "paragraph": "1",
                "point": None,
                "subpoint": None,
                "section": "Transparency obligations",
                "content": "Providers shall inform natural persons when they interact with an AI system.",
            },
            {
                "document_id": 1,
                "original_name": "EU-AI-Act-2024-1689-EN.pdf",
                "article": "Article 5",
                "paragraph": "1",
                "point": "g",
                "subpoint": None,
                "section": "Prohibited AI practices",
                "content": "Biometric categorisation for sensitive traits is prohibited; this prohibition has limited exceptions.",
            },
            {
                "document_id": 2,
                "original_name": "GDPR-2016-679-EN.pdf",
                "article": "Article 9",
                "paragraph": "1",
                "point": None,
                "subpoint": None,
                "section": "Processing of special categories of personal data",
                "content": "Processing biometric data for uniquely identifying a natural person shall be prohibited.",
            },
        ]
        for chunk_index, chunk in zip((3, 4, 1, 0), chunks, strict=True):
            chunk["chunk_index"] = chunk_index

        selected = _comparison_clause_indices(
            "How do the GDPR and the AI Act regulate biometric categorisation?",
            chunks,
            anchor_index=0,
            ranking_scores=np.asarray([0.90, 0.88, 0.85, 0.70], dtype=np.float32),
        )

        self.assertEqual({chunks[index]["article"] for index in selected}, {"Article 5", "Article 50"})
        self.assertEqual(selected, [2, 0])


if __name__ == "__main__":
    unittest.main()
