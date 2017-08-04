import re
import sys

import json
import requests
import datetime as dt

import nltk
from nltk import Tree
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer


def get_query(url, params):
    """
    Gets the raw search query from the landing page url. 

    Params:
    -------
    url: str
        The page from where to submit a GET request for the search query

    params: dict
        A list of key value pairs with which to search the query from. 
        For example, if we want to search for the keyword 'nlp' from google we 
        would pass in the (key, value) pairs of {'site': None, 'source': 'hp', 
        'q': 'nlp'} as the search parameters. 

    Returns:
    --------
    r: Request object
        The request object from which we can get the raw text returned by the GET request. 

    Example:
    --------
    >>> import requests
    >>> url = 'www.google.com/search'
    >>> params = {'site': None, 'source': 'hp', 'q': 'nlp'}
    >>> r = get_query(url, params)
    """
    # Specify the header information
    header = {'user-agent': ('Mozilla/5.0 (Windows NT 10.0; WOW64)'
                          'AppleWebKit/537.36 (KHTML, like Gecko)'
                          'Chrome/57.0.2987.133 Safari/537.36'),
              'referer': None,
              'connection':'keep-alive'}
    r = requests.get(url, headers=header, params=params)

    return r


def clean(raw, get_tags=True):
    """
    Removes unnecssary words and punctuation from the raw string.

    Params:
    -------
    raw: str
        The raw, unfiltered string.

    get_tags: bool, optional, default=True
        Whether or not to get the part of speech tags for the filtered words

    Returns:
    --------
    filtered_sentence: list of str
        The string, removing all unecessary words.
    """
    # Tokenize and remove unimportant punctuations
    words = re.sub(r'[^\w\s]', '', raw)
    filtered_sentence = word_tokenize(words)

    if get_tags:
        filtered_sentence = nltk.pos_tag(filtered_sentence)

    return filtered_sentence

    
def get_origin_and_destination(tagged_words):
    """
    Gets the origin and destination of where the user wants to travel. 

    Params:
    -------
    tagged_words: list of tuples
        The tokenized and tagged words of the string based on the pre-trained UPenn corpus.

    Returns:
    --------
    parsed_words: Tree object
        The parsed string of words into a tree defining the new struture of the tagged string. 
        The origin and destination are organized into one list.
    """
    # Define the expression to get both origin and destination. Usually both the 
    # origin and destination are proper nouns (NNP) with the <TO> signifying the direction of travel. 
    grams = r"Origin/Destination: {<NNP.?>+<TO><NNP?>*}"
    parser = nltk.RegexpParser(grams)

    parsed_tree = parser.parse(tagged_words)

    return parsed_tree


def get_dates(tagged_words):
    """
    Gets the dates of departure and return. 

    Params:
    -------
    tagged_words: list of tuples
        The tokenized and tagged words of the string based on the pre-trained UPenn corpus.

    Returns:
    --------
    parsed_words: Tree object
        The parsed string of words into a tree defining the new struture of the tagged string. 
        The departure and return dates are organized into one list.

    TODO:
    -----
    - There may be scenarios where no date or month is provided, so we will have to 
        account for that scenario.
    - When months are given in lowercase/abbreviated form, the nltk tags result in them 
        being classified as JJ (aka adjective) or RB (aka adverb). For example, the string:

            raw = "Flights from NYC to LAX from oct 3 or 7 till november 11"
        
        is tagged as: 
        [('Flights', 'NNS'), ('from', 'IN'), ('NYC', 'NNP'), ('to', 'TO'), ('LAX', 'VB'), 
         ('from', 'IN'), ('october', 'JJ'), ('3', 'CD'), ('or', 'CC'), ('7', 'CD'), 
         ('till', 'JJ'), ('november', 'RB'), ('11', 'CD')]

        This is a issue as it will not get parsed out properly.
    """
    grams = r"Departure/Return: {<NNP.?>+<CD><TO>*<NNP?>*<CD>*}"
    parser = nltk.RegexpParser(grams)

    parsed_tree = parser.parse(tagged_words)

    # Check if there exists a nested tree object named "Departure/Return"
    for subtree in parsed_tree:
        try:
            if "Departure/Return" in subtree.label():
                # Since there will be a min of 2 words but at most 5 words in 
                # this subtree, we can simply return the departure date and 
                # check if there is a return date
                departure_date = subtree[0][0] + " " + subtree[1][0]
                if len(subtree) < 5:
                    return_date = None
                else:
                    return_date = subtree[3][0] + " " + subtree[4][0]
        except AttributeError:
            continue

    return departure_date, return_date


def convert_date(date):
    """Temporary method, will move."""
    return dt.datetime.strptime(date, "%b %d")


def clean_date(date):
    """
    Cleans the string formatting of date to get it to a unified format of:
    <Month> <Date> <Year> (i.e. Dec 1 2017).

    Input Examples: 
        1. Dec
        2. December
        3. Dec 1st 
        4. December 1st
        5. Dec 2017
        6. December 2017
        5. Dec 1st 2017
        6. December 1st 2017

    Params:
    -------
    date: str
        The date as expressed in human readable format

    Returns:
    --------
    cleaned_date: str
        The date in the specified format of <Month> <Date> <Year>
    """
    # Split to get individual parts of the date
    date = date.split()

    raise NotImplementedError



def is_flexible(tagged_words):
    """
    Checks whether or not the user has requested for a fexible fare (i.e. variable 
    dates and times). 

    Params:
    -------
    tagged_words: list of tuples
        The tokenized and tagged words of the string based on the pre-trained UPenn corpus.

    Returns:
    --------
    bool: 
        Whether or not the searched flight request has flexible dates or not.
    """
    # Check if the keyword "flexible" is within the input. 
    for word in tagged_words:
        if ("Flexible" in word) or ("flexible" in word):
            return True

    # Check for any indication of flexibles dates (i.e. October 8 or October 11 to XXXX)
    grams = r"Flexible: {<CD.?>+<CC><CD.?>}"
    parser = nltk.RegexpParser(grams)

    parsed_tree = parser.parse(tagged_words)

    # Check if there exists a nested tree object named "Flexible"
    for subtree in parsed_tree:
        try: 
            if ("Flexible" in subtree.label()): 
                return True
        except AttributeError:
            continue
    return False


def tree_to_json(tree):
    """
    Converts a nltk tree to a JSON object. In reality, it is just a useful 
    abstraction of the json.dump() method. 

    Params:
    -------
    tree: nltk Tree object
        The tree output returned from running a search query and extracting the
        necessary information.

    Returns:
    --------
    json: JSON object
        The tree/dictionary converted to its JSON form. 
    """
    tdict = _tree_to_dict(tree)
    return json.dump(tdict, sys.stdout, indent=2)


def _tree_to_dict(tree):
    """
    Internal function, to be used by tree_to_json to build json output.

    Params:
    -------
    tree: nltk Tree object
        The tree output returned from running a search query and extracting the
        necessary information.

    Returns:
    --------
    dict: dict
        The parsed words tree converted into its (key, value) pairs dictionary, 
        taking into account nested trees. 
    """
    return {tree.label(): [tree_to_dict(t) if isinstance(t, Tree) else t for t in tree]}
