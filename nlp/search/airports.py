import os
import requests


API_KEY = os.environ['IATA_API_KEY']


def is_airport_code(location):
    """
    Determines whether or not the location (i.e. an airport name, city, or etc.) 
    provided is part of the airport codes defined by the International Air 
    Transport Association (IATA).

    One criteria for IATA airport code is that they must be only 3 letters (all 
    CAPS). For example, JFK is an airport code, but SUEG would not be. 

    Params:
    -------
    location: str
        The unique location identifier (airport name, city, etc.). 

    Returns:
    --------
    bool:
        Whether or not the location is an IATA airport code. 
    """
    # Remove trailing and leading white spaces, if any
    location = location.strip()

    # Check if the location is an airport code within the database
    url = 'http://www.iatacodes.org/api/v6/airports'
    params = {'api_key': API_KEY, 'code': location}
    r = requests.get(url, params=params)

    return len(r.json()['response']) > 0


def get_airport_code(location):
    """
    Returns the IATA airport code associated with the given location.

    Params:
    -------
    location: str
        The unique location identifier (airport name, city, etc.). 

    Returns:
    --------
    airport_code: str
        The 3-letter airport code (i.e. JFK, CDG) associated with the given location. 
    """
    # Remove trailing and leading white spaces, if any
    location = location.strip()

    # If the provided location is an airport code already
    if is_airport_code(location):
        return location.upper() # incase input is lowercase
    else:
        url = 'http://www.iatacodes.org/api/v6/autocomplete'
        params = {'api_key': API_KEY, 'query': location}
        r = requests.get(url, params=params)

        # Get the top airport within the country, then by city
        country_airports = r.json()['response']['airports_by_countries']
        city_airports = r.json()['response']['airports_by_cities']
        if len(country_airports) > 0:
            airport_code = country_airports[0]['code']
        elif len(city_airports) > 0:
            airport_code = city_airports[0]['code']
        else:
            airport_code = None
        
        return airport_code
