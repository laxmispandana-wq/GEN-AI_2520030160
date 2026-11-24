import unittest

from llm_client import LLMClient


class FakeProvider:

    def generate(self, prompt):
        return "Fake response"


class FakeLogger:

    def __init__(self):
        self.records = []

    def log_call(self, record):
        self.records.append(record)


class TestLLMClient(unittest.TestCase):

    def test_logger_records_call(self):

        logger = FakeLogger()

        record = {
            "provider": "test",
            "prompt": "Hello",
            "response": "Fake response",
            "status": "success"
        }

        logger.log_call(record)

        self.assertEqual(len(logger.records), 1)
        self.assertEqual(logger.records[0]["status"], "success")


if __name__ == "__main__":
    unittest.main()