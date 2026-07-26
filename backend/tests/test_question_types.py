import unittest

from app.services.retrieval.questions import QuestionType, classify_question_type
from app.services.retrieval import _definition_scores, _question_type_adjustment


class QuestionTypeTests(unittest.TestCase):
    def test_supported_question_types(self) -> None:
        cases = {
            "What is an AI system?": QuestionType.DEFINITION,
            "What is this project about?": QuestionType.SUMMARY,
            "What information must a notification contain?": QuestionType.CONTENT,
            "When must the controller notify the data subject?": QuestionType.CONDITION,
            "What practices are prohibited?": QuestionType.OBLIGATION,
            "List the required documents.": QuestionType.LIST,
            "How do the GDPR and the AI Act address automated decisions?": QuestionType.COMPARISON,
            "How do I install the application?": QuestionType.PROCEDURE,
            "Summarise the installation section.": QuestionType.SUMMARY,
        }

        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(classify_question_type(question), expected)

    def test_definition_matching_requires_the_exact_subject_at_clause_start(self) -> None:
        scores = _definition_scores(
            "What is an AI system?",
            [
                "(1) ‘AI system’ means a machine-based system.",
                "(18) ‘performance of an AI system’ means its ability to achieve a purpose.",
            ],
        )

        self.assertEqual(scores.tolist(), [1.0, 0.0])

    def test_obligation_question_prefers_a_direct_rule_over_a_reference_clause(self) -> None:
        question = "What are the transparency obligations for certain AI systems?"

        direct_rule = _question_type_adjustment(
            question,
            "Providers shall inform natural persons that they are interacting with an AI system.",
        )
        reference_clause = _question_type_adjustment(
            question,
            "6. Paragraphs 1 to 4 shall not affect the requirements set out in Chapter III.",
        )

        self.assertGreater(direct_rule, reference_clause)

    def test_content_question_prefers_a_direct_requirement_over_an_indirect_reference(self) -> None:
        question = "What information must a personal data breach notification contain?"

        direct_rule = _question_type_adjustment(
            question,
            "The notification shall at least contain the following information.",
        )
        indirect_reference = _question_type_adjustment(
            question,
            "The communication shall contain the information referred to in Article 33.",
        )

        self.assertGreater(direct_rule, indirect_reference)

    def test_comparison_prefers_an_operative_rule_over_a_definition(self) -> None:
        question = "How do Policy A and Policy B regulate biometric categorisation?"

        definition = _question_type_adjustment(
            question,
            "(1) ‘biometric categorisation system’ means an AI system that assigns categories.",
        )
        prohibition = _question_type_adjustment(
            question,
            "The use of biometric categorisation systems is prohibited; this prohibition has a limited exception.",
        )

        self.assertGreater(prohibition, definition)

    def test_condition_question_preserves_actor_action_target_direction(self) -> None:
        question = "When should a controller notify the supervisory authority?"

        direct_rule = _question_type_adjustment(
            question,
            "In the case of a personal data breach, the controller shall notify "
            "the supervisory authority without undue delay.",
        )
        reversed_rule = _question_type_adjustment(
            question,
            "The supervisory authority shall notify the complainant and inform "
            "the controller of its decision.",
        )

        self.assertGreater(direct_rule, reversed_rule)


if __name__ == "__main__":
    unittest.main()
