"""
Natural Language Processing (NLP) search engine for extracting information from
search queries that ask about airfare tickets.

Created by Ayush Karnawat on 8/15/2017.
"""

import re
import json
import requests
import datetime as dt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nlp.search import airports


class Search(object):
    """
    Basic NLP search engine for parsing information from search queies about airfare tickets. 
    Parses relevant information from a raw string to a structured JSON output.

    Params:
    -------
    raw: str
        The raw, unfiltered string.
    """


    def __init__(self, raw):
        self.raw = raw
        self.tagged_words = self.clean(get_tags=True)
        self.origin_city, self.destination_city = self.get_origin_and_destination()
        self.departure_date, self.return_date = self.get_dates()


    def clean(self, get_tags=True):
        """
        Removes unnecssary words and punctuations from the raw string.

        Params:
        -------
        get_tags: bool, optional, default=True
            Whether or not to get the part of speech tags for the filtered words

        Returns:
        --------
        filtered_sentence: list of str
            The string, removing all unecessary words and punctuations.
        """
        # Tokenize and remove unimportant punctuations
        # TODO: This removes hyphens as well, which might be useful
        sentence = re.sub(r'[^\w\s]', '', self.raw)
        words = word_tokenize(sentence)

        # Only capatalize the important words
        filtered_sentence = []
        stop_words = set(stopwords.words('english'))
        for word in words:
            if word not in stop_words:
                filtered_sentence.append(word.title())
            else:
                filtered_sentence.append(word)

        # Tag all words with their part of speech
        if get_tags:
            filtered_sentence = nltk.pos_tag(filtered_sentence)

        return filtered_sentence


    def is_flexible(self):
        """
        Checks whether or not the user has requested for a flexible flight (i.e. 
        variable dates and times). 

        Returns:
        --------
        bool: 
            Whether or not the searched flight request has flexible dates or not.
        """
        # Check if the keyword "flexible" is within the input.
        for word in self.tagged_words:
            if ("Flexible" in word) or ("flexible" in word):
                return True

        # Check for any indication of flexibles dates (i.e. October 8 or
        # October 11 to XXXX)
        grams = r"Flexible: {<CD.?>+<CC><CD.?>}"
        parser = nltk.RegexpParser(grams)

        parsed_tree = parser.parse(self.tagged_words)

        # Check if there exists a nested tree object named "Flexible"
        for subtree in parsed_tree:
            try:
                if ("Flexible" in subtree.label()):
                    return True
            except AttributeError:
                continue
        return False


    def get_origin_and_destination(self):
        """
        Gets the origin and destination of where the user wants to travel. 

        Params:
        -------
        tagged_words: list of tuples
            The tokenized and tagged words of the string based on the pre-trained UPenn corpus.

        Returns:
        --------
        origin_city: str
            The 3-letter IATA code representaion of the origin city.

        destination_city: str
            The 3-letter IATA code representation of the destination city.
        """
        # Define the expression to get both origin and destination. Usually both the 
        # origin and destination are proper nouns (NNP) with the <TO> signifying the direction of travel. 
        grams = r"Origin/Destination: {<NNP.?>+<TO><NNP?>*}"
        parser = nltk.RegexpParser(grams)

        parsed_tree = parser.parse(self.tagged_words)

        # Check if there exists a nested tree object named "Departure/Return"
        for subtree in parsed_tree:
            try:
                if "Origin/Destination" in subtree.label():
                    # A city name can be arbitrarily long, so we will have to use 
                    # another tree parser have to get the name until the "to" keyword
                    origin_grams = r"Origin: {<NNP.?>+<TO>}"
                    origin_parser = nltk.RegexpParser(origin_grams)
                    nested_subtree = origin_parser.parse(subtree)

                    # Get the origin city
                    origin_city = ""
                    for i in range(0, len(nested_subtree[0]) - 1):
                            origin_city += nested_subtree[0][i][0] + " "

                    # Get the departure city
                    destination_city = ""
                    for word in nested_subtree[1:]:
                        destination_city += word[0] + " "
            except AttributeError:
                continue

        # Get airport codes associated with the cities
        origin_city = airports.get_airport_code(origin_city)
        destination_city = airports.get_airport_code(destination_city)

        return origin_city, destination_city


    def get_dates(self):
        """
        Gets the dates of departure and return. 

        Params:
        -------
        tagged_words: list of tuples
            The tokenized and tagged words of the string based on the pre-trained UPenn corpus.

        Returns:
        --------
        departure_date: int
            The UNIX timestamp (in milleseconds) of the departure date. 

        return_date: int
            The UNIX timestamp (in milliseconds) of the return date, if available.

        TODO:
        -----
        - When months are given in lowercase/abbreviated form, the nltk tags result in them 
            being classified as JJ (aka adjective) or RB (aka adverb). For example, the string:

                raw = "Flights from NYC to LAX from oct 3 or 7 till november 11"
            
            is tagged as: 
            [('Flights', 'NNS'), ('from', 'IN'), ('NYC', 'NNP'), ('to', 'TO'), ('LAX', 'VB'), 
            ('from', 'IN'), ('october', 'JJ'), ('3', 'CD'), ('or', 'CC'), ('7', 'CD'), 
            ('till', 'JJ'), ('november', 'RB'), ('11', 'CD')]

            This is a issue as it will not get parsed out properly.
        """
        grams = r"Departure/Return: {<NNP.?>*<CD>+<TO>*<NNP.?>*<CD>*}"
        parser = nltk.RegexpParser(grams)

        parsed_tree = parser.parse(self.tagged_words)

        # Check if there exists a nested tree object named "Departure/Return"
        for subtree in parsed_tree:
            try:
                if "Departure/Return" in subtree.label():
                    # Since there will be a min of 2 words but at most 5 words in 
                    # this subtree, we can simply return the departure date and 
                    # check if there is a return date
                    departure_date = subtree[0][0] + " " + subtree[1][0]
                    if len(subtree) < 3:
                        return_date = None
                    else:
                        return_date = subtree[3][0] + " " + subtree[4][0]
            except AttributeError:
                continue

        # Clean and convert the dates to UNIX datetime stamp with millisecond percision
        departure_date = int(dt.datetime.strptime(format_date(departure_date), "%b %d %Y").timestamp() * 1e3)
        if return_date is not None:
            return_date = int(dt.datetime.strptime(format_date(return_date), "%b %d %Y").timestamp() * 1e3)

        return departure_date, return_date

    
    def to_json(self):
        """
        Convert the relevant information to JSON.

        Returns:
        --------
        out: JSON object
            The information in its JSON format.
        """
        # Build output
        output = {
            'request': self.raw,
            'response': {
                'origin': self.origin_city,
                'destination': self.destination_city,
                'departure': self.departure_date,
                'return': self.return_date
            }
        }
        return json.dumps(output)


def format_date(date):
    """
    Cleans the string formatting of date to get it to a unified format of:
    <Month> <Date> <Year> (i.e. Dec 1 2017).

    Params:
    -------
    date: str
        The date as expressed in human readable format.

    Returns:
    --------
    formatted_date: str
        The date in the specified format of <Month> <Date> <Year>.

    Examples:
    ---------
    >>> date = "April 13th 2017"
    "Apr 13 2017"
    >>> date = "December 15th"
    "Dec 15 2017"
    >>> date = "December"
    "Dec 1 2017"
    >>> date = "2017" 
    "Aug(or whatever the current month is) 1 2017"
    """
    # Split to get individual parts of the date
    date = date.split()

    month = None
    day = None
    year = None
    for word in date:
        # If the current word is the month
        if len(re.findall(r'^\b[A-Za-z]+$', word)) > 0:
            if len(word) > 3: # if not in abbreviated form, abbreviate it
                month = word[:3].title()
            else:
                month = word.title()

        # If the word is the date (with or without the 'st, nd, th' parts)
        if len(re.findall(r'^\d{1,2}[st|St|nd|Nd|rd|Rd|th|Th]*$', word)) > 0:
            if len(word) > 2: # Remove the "st, nd, th" parts of the date
                day = word[:-2]
            else:
                day = word

        # If the word is the year
        if len(re.findall(r'^\d{4}$', word)) > 0:
            year = word

    # Default values 
    now = dt.datetime.now()
    if month is None:
        month = now.strftime("%b")
    if day is None:
        day = now.strftime("%d")
    if year is None:
        year = now.strftime("%Y")

    return month + " " + day + " " + year


if __name__ == "__main__":
    raw = "Flights from New York to DEL on October 2nd"
    s = Search(raw)
    print(s.tagged_words)
    print(s.origin_city)
    print(s.destination_city)
    print(s.departure_date)
    print(s.return_date)
    print(s.to_json())