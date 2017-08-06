import unittest

from nlp.search import search


class TestNLPSearch(unittest.TestCase):

    def test_search(self):
        # Test case #1: Test for origin and and destination based on airport codes
        raw = "Flights from JFK to LAX between October 2nd to November 21st"
        output = search.process(raw)
        output_airport_code = """{
    "request": "Flights from JFK to LAX between October 2nd to November 21st",
    "response": {
        "origin": "JFK",
        "destination": "LAX",
        "departure": 1506916800.0,
        "return": 1511240400.0
    }
}"""
        self.assertEqual(output, output_airport_code)

        # Test case #2: Test for origin and destination based on city name
        raw = "Flights from New York to Los Angeles between October 2nd to November 21st"
        output = search.process(raw)
        output_city_names = """{
    "request": "Flights from New York to Los Angeles between October 2nd to November 21st",
    "response": {
        "origin": "JFK",
        "destination": "LAX",
        "departure": 1506916800.0,
        "return": 1511240400.0
    }
}"""
        self.assertEqual(output, output_city_names)

        # Test case #3: Test for origin and destination based on one with city name and one airport code
        raw = "Flights from New York to LAX between October 2nd to November 21st"
        output = search.process(raw)
        output_city_code = """{
    "request": "Flights from New York to LAX between October 2nd to November 21st",
    "response": {
        "origin": "JFK",
        "destination": "LAX",
        "departure": 1506916800.0,
        "return": 1511240400.0
    }
}"""
        self.assertEqual(output, output_city_code)

        # Test case #4: Both departure and return date are provided
        raw = "Flights from New York to CDG from October 2nd to November 21st"
        output = search.process(raw)
        output_departure_return_dates = """{
    "request": "Flights from New York to CDG from October 2nd to November 21st",
    "response": {
        "origin": "JFK",
        "destination": "CDG",
        "departure": 1506916800.0,
        "return": 1511240400.0
    }
}"""
        self.assertEqual(output, output_departure_return_dates)

        # Test case #5: Only departure date is provided
        raw = "Flights from New York to DEL on October 2nd"
        output = search.process(raw)
        output_departure_date = """{
    "request": "Flights from New York to DEL on October 2nd",
    "response": {
        "origin": "JFK",
        "destination": "DEL",
        "departure": 1506916800.0,
        "return": null
    }
}"""
        self.assertEqual(output, output_departure_date)


if __name__ == '__main__':
    unittest.main()
