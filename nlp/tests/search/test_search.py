import json
import unittest

from nlp.search.search import Search


class TestSearch(unittest.TestCase):

    def test(self):
        # Load all test cases and test
        with open("nlp/test/search/search_test_cases.json") as json_file:
            test_cases = json.load(json_file)

        for test_case in test_cases:
            raw = test_case['request']
            output = Search(raw).to_json()
            response = json.dumps(test_case) # Convert to str to compare outputs
            self.assertEqual(output, response)

if __name__ == '__main__':
    unittest.main()
