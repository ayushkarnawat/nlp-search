import unittest

import nltk
from nltk.tokenize import word_tokenize

from nlp.search import search


class TestNLPSearch(unittest.TestCase):

    def test_filter(self):
        # Test case #1: Untagged words
        raw = "Flights from New York to Los Angeles between October 2nd to November 21st"
        filtered_sentence_without_tags = search.filter(raw, get_tags=False, language="english")
        result = ['Flights', 'New', 'York', 'Los', 'Angeles', 'October', '2nd', 'November', '21st']
        self.assertEquals(filtered_sentence_without_tags, result)

        # Test case #2: Proper word string tagging based on parts of speech
        raw = "Flights from New York to Los Angeles between October 2nd to November 21st"
        filtered_sentence_with_tags = search.filter(raw, get_tags=True, language="english")
        result_tagged = [('Flights', 'NNS'), ('New', 'NNP'), ('York', 'NNP'), ('Los', 'NNP'), ('Angeles', 'NNP'), ('October', 'NNP'),                      ('2nd', 'CD'), ('November', 'NNP'), ('21st', 'CD')]
        self.assertEquals(filtered_sentence_with_tags, result_tagged)


    def test_get_origin_and_destination(self):
        # Test case #1: Test for origin and and destination based on airport codes
        raw = "Flights from JFK to LAX between October 2nd to November 21st"
        words = word_tokenize(raw)
        tagged_words = nltk.pos_tag(words)
        parsed_tree = search.get_origin_and_destination(tagged_words)
        output_airport_code = "(S\n  Flights/NNS\n  from/IN\n  (ORGIN/DESTINATION JFK/NNP to/TO LAX/NNP)\n  between/IN\n  October/NNP\n  2nd/CD\n  to/TO\n  November/NNP\n  21st/CD)"
        self.assertEquals(parsed_tree.__str__(), output_airport_code)

        # Test case #2: Test for origin and destination based on city name
        raw = "Flights from New York to Los Angeles between October 2nd to November 21st"
        words = word_tokenize(raw)
        tagged_words = nltk.pos_tag(words)
        parsed_tree = search.get_origin_and_destination(tagged_words)
        output_city_names = "(S\n  Flights/NNS\n  from/IN\n  (ORGIN/DESTINATION New/NNP York/NNP to/TO Los/NNP Angeles/NNP)\n  between/IN\n  October/NNP\n  2nd/CD\n  to/TO\n  November/NNP\n  21st/CD)"
        self.assertEquals(parsed_tree.__str__(), output_city_names)

        # Test case #3: Test for origin and destination based on one with city
        # name and one airport code
        raw = "Flights from New York to LAX between October 2nd to November 21st"
        words = word_tokenize(raw)
        tagged_words = nltk.pos_tag(words)
        parsed_tree = search.get_origin_and_destination(tagged_words)
        output_city_code = "(S\n  Flights/NNS\n  from/IN\n  (ORGIN/DESTINATION New/NNP York/NNP to/TO LAX/NNP)\n  between/IN\n  October/NNP\n  2nd/CD\n  to/TO\n  November/NNP\n  21st/CD)"
        self.assertEquals(parsed_tree.__str__(), output_city_code)


    def test_get_dates(self):
        # Test case #1: Both departure and return date are provided
        raw = "Flights from New York to LAX from October 2nd to November 21st"
        words = word_tokenize(raw)
        tagged_words = nltk.pos_tag(words)
        parsed_tree = search.get_dates(tagged_words)
        output_departure_return_dates = "(S\n  Flights/NNS\n  from/IN\n  New/NNP\n  York/NNP\n  to/TO\n  LAX/VB\n  from/IN\n  (Departure/Return October/NNP 2nd/CD to/TO November/NNP 21st/CD))"
        self.assertEquals(parsed_tree.__str__(), output_departure_return_dates)

        # Test case #2: Only departure date is provided
        raw = "Flights from New York to LAX on October 2nd"
        words = word_tokenize(raw)
        tagged_words = nltk.pos_tag(words)
        parsed_tree = search.get_dates(tagged_words)
        output_departure_date = "(S\n  Flights/NNS\n  from/IN\n  New/NNP\n  York/NNP\n  to/TO\n  LAX/VB\n  on/IN\n  (Departure/Return October/NNP 2nd/CD))"
        self.assertEquals(parsed_tree.__str__(), output_departure_date)


if __name__ == '__main__':
    unittest.main()