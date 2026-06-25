import unittest

from main import Dezzy


class DummyQuestion:
    def __init__(self, text: str):
        self.question_text = text


class GroundedPremortemTests(unittest.TestCase):
    def test_build_grounded_context_includes_research_and_premortem(self):
        bot = Dezzy.__new__(Dezzy)
        bot._grounding_instructions = lambda: "Ground every forecast in evidence."

        question = DummyQuestion("Will the project launch by Q4?")
        context = bot._build_grounded_context(question, "Recent product signals", "The plan could fail due to staffing.")

        self.assertIn("Ground every forecast in evidence.", context)
        self.assertIn("Recent product signals", context)
        self.assertIn("The plan could fail due to staffing.", context)
        self.assertIn("Question:", context)


if __name__ == "__main__":
    unittest.main()
