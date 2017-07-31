import re
import sys

import json
import requests

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


def clean(raw, get_tags=True, language="english"):
    """
    Removes unnecssary words from the raw string.

    Params:
    -------
    raw: str
        The raw, unfiltered string.

    get_tags: bool, optional, default=True
        Whether or not to get the part of speech tags for the filtered words

    language: str, optional, default="english"
        The language of the raw string. This will define the dictionary 
        of stop words to remove from the unfiltered string.

    Returns:
    --------
    filtered_sentence: list of str
        The string, removing all unecessary words.
    """
    # Define the stop_words
    stop_words = set(stopwords.words(language))

    # Tokenize and remove unimportant words
    words = word_tokenize(raw)
    filtered_sentence = [w for w in words if not w in stop_words]

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
    grams = r"ORGIN/DESTINATION: {<NNP.?>+<TO><NNP?>*}"
    parser = nltk.RegexpParser(grams)

    parsed_words = parser.parse(tagged_words)

    return parsed_words


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
    """
    grams = r"Departure/Return: {<NNP.?>+<CD><TO>*<NNP?>*<CD>*}"
    parser = nltk.RegexpParser(grams)

    parsed_words = parser.parse(tagged_words)

    return parsed_words


# def is_flexibe():
#     """
#     Checks whether or not the user has requested for a fexible fare (i.e. variable 
#     dates and times). 
#     """


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
    tdict = {tree.label(): [tree_to_dict(t) if isinstance(t, Tree) else t for t in tree]}
    return json.dump(tdict, sys.stdout, indent=2)
