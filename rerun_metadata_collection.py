# standard library imports
from datetime import datetime, timezone
import json
import logging
import os
import re
import sys
import time

# third-party imports
from dotenv import load_dotenv
import polars as pl
import psycopg2
import requests
import yaml

# local/custom imports
from data_models.training_data_frame import TrainingDataFrame

# ------------------------------------------------------------------------------
# initialization and setup
# ------------------------------------------------------------------------------

# logger config
logger = logging.getLogger(__name__)

# if not inherited set parameters here
if __name__ == "__main__" or not logger.handlers:
    # Set up standalone logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    logging.getLogger("paramiko").setLevel(logging.INFO)
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

# Load environment variables from .env file
load_dotenv(override=True)

# pipeline env vars
batch_size = int(os.getenv('BATCH_SIZE', 50))

# load api env vars
movie_details_api_base_url = os.getenv('MOVIE_DETAILS_API_BASE_URL')
movie_ratings_api_base_url = os.getenv('MOVIE_RATINGS_API_BASE_URL')
movie_details_api_key = os.getenv('MOVIE_DETAILS_API_KEY')
movie_ratings_api_key = os.getenv('MOVIE_RATINGS_API_KEY')

tv_details_api_base_url = os.getenv('TV_DETAILS_API_BASE_URL')
tv_ratings_api_base_url = os.getenv('TV_RATINGS_API_BASE_URL')
tv_details_api_key = os.getenv('TV_DETAILS_API_KEY')
tv_ratings_api_key = os.getenv('TV_RATINGS_API_KEY')

# ------------------------------------------------------------------------------
# database functions
# ------------------------------------------------------------------------------

def create_conn(env: str) -> psycopg2.extensions.connection:
    """
    created connection object with given items from the connection yaml based
        off of the env provided

    :param env: env name as a string to connect to
    :return: psycopg2 connection based off of the env provided
    """
    # Read the YAML configuration file
    try:
        with open('.environments.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError("environments.yaml file not found")

    # Get the database configuration for the specified environment
    if env not in config['pgsql']:
        raise ValueError(f"Environment '{env}' not found in configuration")

    db_config = config['pgsql'][env]

    # Create the connection string
    conn_string = f"host={db_config['endpoint']} " \
                  f"port={db_config['port']} " \
                  f"dbname={db_config['database_name']} " \
                  f"user={db_config['username']} " \
                  f"password={db_config['password']}"

    # Connect to the database
    conn = psycopg2.connect(conn_string)

    # Set the schema as the default for lookups
    with conn.cursor() as cursor:
        cursor.execute(f"SET search_path TO {db_config['schema']}")
        conn.commit()

    return conn


def get_training_data(conn: psycopg2.extensions.connection) -> TrainingDataFrame:
    """
    Retrieve all training data from the database

    :param conn: database connection
    :return: TrainingDataFrame containing all training data
    """
    query = "SELECT * FROM training ORDER BY imdb_id"

    with conn.cursor() as cursor:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

    if not rows:
        return None

    # Convert to polars DataFrame
    df = pl.DataFrame(rows, schema=columns, orient='row', infer_schema_length=None)
    return TrainingDataFrame(df)


def update_training_batch(conn: psycopg2.extensions.connection, training_batch: TrainingDataFrame):
   """
   Update training records in the database for a batch

   :param conn: database connection
   :param training_batch: TrainingDataFrame containing updated records
   """
   # Protected columns that should not be updated
   protected_columns = {'label', 'imdb_id', 'tmdb_id', 'human_labeled', 'anomalous', 'created_at', 'updated_at'}

   # Get all columns except protected ones
   updatable_columns = [col for col in training_batch.df.columns if col not in protected_columns]

   # Build the UPDATE query
   set_clause = ", ".join([f"{col} = %s" for col in updatable_columns])
   query = f"""
       UPDATE training 
       SET {set_clause}
       WHERE imdb_id = %s
   """

   # Convert all polars nulls to None for database compatibility
   records = []
   for row in training_batch.df.iter_rows(named=True):
       # Replace polars.Null with None in each row
       clean_row = {k: (None if v is None or str(v) == "None" else v) for k, v in row.items()}
       records.append(clean_row)

   with conn.cursor() as cursor:
       for row in records:
           try:
               # Prepare values for update (excluding protected columns)
               values = []
               for col in updatable_columns:
                   value = row[col]

                   # Handle None values and empty lists
                   if value is None:
                       values.append(None)
                   elif isinstance(value, list):
                       if len(value) == 0:
                           values.append(None)
                       else:
                           values.append(value)
                   else:
                       values.append(value)

               # Add imdb_id for WHERE clause
               values.append(row['imdb_id'])

               cursor.execute(query, values)

           except Exception as e:
               logging.error(f"Failed to update record {row['imdb_id']}: {e}")
               # Rollback the current transaction and start a new one
               conn.rollback()
               continue

   try:
       conn.commit()
       logging.debug(f"Successfully updated {training_batch.df.height} records")
   except Exception as e:
       logging.error(f"Failed to commit batch: {e}")
       conn.rollback()


# ------------------------------------------------------------------------------
# metadata collection helper functions
# ------------------------------------------------------------------------------

def collect_details(media_item: dict) -> dict:
    """
    uses TMDB to get the details of a media item; this is a different API
        then the one above, which accepts only the tmdb_id collected above

    :param media_item: dict containing one row of training data
    :return: dict of items with metadata added
    """
    response = {}

    # prepare and send response
    if media_item['media_type'] == 'movie':
        params = {'api_key': movie_details_api_key}
        url = f"{movie_details_api_base_url}/{media_item['tmdb_id']}"
        logging.debug(f"collecting metadata details for: {media_item['imdb_id']}")
        response = requests.get(url, params=params)
    elif media_item['media_type'] in ['tv_show', 'tv_season']:
        params = {'api_key': tv_details_api_key}
        url = f"{tv_details_api_base_url}/{media_item['tmdb_id']}"
        logging.debug(f"collecting metadata details for: {media_item['imdb_id']}")
        response = requests.get(url, params=params)

    # verify successful API response
    if response.status_code != 200:
        logging.error(f"media details API returned status code: {response.status_code} for {media_item['imdb_id']}")
        return media_item

    # if no issue load results
    data = json.loads(response.content)

    # collect time information
    year_pattern = r'(19|20)\d{2}'
    if media_item['media_type'] == 'movie':
        if 'release_date' in data and data['release_date']:
            release_year_match = re.search(year_pattern, data.get('release_date'))
            if release_year_match:
                media_item['release_year'] = int(release_year_match[0])
    elif media_item['media_type'] in ['tv_show', 'tv_season']:
        if 'first_air_date' in data and data['first_air_date']:
            release_year_match = re.search(year_pattern, data.get('first_air_date'))
            if release_year_match:
                media_item['release_year'] = int(release_year_match[0])

    # collect quantitative fields
    if 'budget' in data:
        media_item['budget'] = data['budget']
    if 'revenue' in data:
        media_item['revenue'] = data['revenue']
    if 'runtime' in data:
        media_item['runtime'] = data['runtime']

    # collect country and production information
    if 'origin_country' in data:
        origin_country = []
        for country in data['origin_country']:
            origin_country.append(country)
        media_item['origin_country'] = origin_country

    if 'production_companies' in data:
        production_companies = []
        for company in data['production_companies']:
            production_companies.append(company['name'])
        media_item['production_companies'] = production_companies

    if 'production_countries' in data:
        production_countries = []
        for country in data['production_countries']:
            production_countries.append(country['iso_3166_1'])
        media_item['production_countries'] = production_countries

    if 'status' in data:
        media_item['production_status'] = data['status']

    # collect language information
    if 'original_language' in data:
        media_item['original_language'] = data['original_language']

    if 'spoken_languages' in data:
        spoken_languages = []
        for language in data.get('spoken_languages'):
            spoken_languages.append(language['iso_639_1'])
        media_item['spoken_languages'] = spoken_languages

    # collect other string fields
    if 'genres' in data:
        genres = []
        for genre in data.get('genres'):
            genres.append(genre['name'])
        media_item['genre'] = genres

    if 'original_title' in data:
        media_item['original_media_title'] = data['original_title']
    elif 'original_name' in data:  # For TV shows
        media_item['original_media_title'] = data['original_name']

    # collect long strings
    if 'overview' in data:
        media_item['overview'] = data['overview']

    if 'tagline' in data:
        media_item['tagline'] = data['tagline']

    # collect TMDB ratings
    if 'vote_average' in data:
        media_item['tmdb_rating'] = data['vote_average']
    if 'vote_count' in data:
        media_item['tmdb_votes'] = data['vote_count']

    return media_item


def collect_ratings(media_item: dict) -> dict:
    """
    get ratings specific details for each media item, e.g. rt_score, metascore

    :param media_item: dict containing one row of training data
    :return: dict of items with metadata added
    """
    response = {}

    # Define the parameters for the OMDb API request
    if media_item['media_type'] == 'movie':
        # if available query by imdb_id
        if media_item['imdb_id'] is not None:
            params = {
                'i': media_item["imdb_id"],
                'apikey': movie_ratings_api_key
            }
        else:
            params = {
                't': media_item["media_title"],
                'apikey': movie_ratings_api_key
            }
            if media_item['release_year'] is not None:
                params['y'] = media_item["release_year"]

        logging.debug(f"collecting ratings for: {media_item['imdb_id']}")
        response = requests.get(movie_ratings_api_base_url, params=params)

    elif media_item['media_type'] in ['tv_show', 'tv_season']:
        # if available query by imdb_id
        if media_item['imdb_id'] is not None:
            params = {
                'i': media_item["imdb_id"],
                'apikey': tv_ratings_api_key
            }
        else:
            params = {
                't': media_item["media_title"],
                'apikey': tv_ratings_api_key
            }
            if media_item['release_year'] is not None:
                params['y'] = media_item["release_year"]

        logging.debug(f"collecting ratings for: {media_item['imdb_id']}")
        response = requests.get(tv_ratings_api_base_url, params=params)

    if response.status_code != 200:
        logging.error(f"media ratings API returned status code: {response.status_code} for {media_item['imdb_id']}")
        return media_item

    data = json.loads(response.content)

    # check if the response was successful, and if so move on
    if data.get("Response") == "True":
        # Extract the metadata from the response
        if data:
            # items to collect for movies and tv shows
            if data.get('Metascore', 'N/A') != "N/A":
                try:
                    media_item['metascore'] = int(data.get('Metascore'))
                except (ValueError, TypeError):
                    media_item['metascore'] = None

            if data.get('imdbRating', 'N/A') != "N/A":
                try:
                    media_item['imdb_rating'] = float(data.get('imdbRating'))
                except (ValueError, TypeError):
                    media_item['imdb_rating'] = None

            if data.get('imdbVotes', 'N/A') != "N/A":
                try:
                    media_item['imdb_votes'] = int(re.sub(r"\D", "", data.get('imdbVotes')))
                except (ValueError, TypeError):
                    media_item['imdb_votes'] = None

            # items to collect only for movies
            if media_item['media_type'] == 'movie':
                if "Ratings" in data:
                    # determine if Rotten tomato exists in json
                    for rating in data.get("Ratings", []):
                        if rating["Source"] == "Rotten Tomatoes":
                            try:
                                media_item['rt_score'] = int(rating["Value"].rstrip('%'))
                            except (ValueError, TypeError):
                                media_item['rt_score'] = None

    # return the updated media_item
    return media_item


# ------------------------------------------------------------------------------
# full metadata collection pipeline
# ------------------------------------------------------------------------------

def collect_metadata(env: str = 'dev'):
    """
    Collect metadata for all training records

    :param env: environment to connect to
    :debug: env='prod'
    :debug: batch=0
    """
    # create database connection
    conn = create_conn(env)

    try:
        # read in existing data
        training = get_training_data(conn)

        # if no training data to process, return
        if training is None:
            logging.info("No training data found to process")
            return

        # batch up operations to avoid API rate limiting
        number_of_batches = (training.df.height + batch_size - 1) // batch_size

        for batch in range(number_of_batches):
            logging.debug(f"starting metadata collection batch {batch+1}/{number_of_batches}")

            # set batch indices
            batch_start_index = batch * batch_size
            batch_end_index = min((batch + 1) * batch_size, training.df.height)

            # create training batch as proper TrainingDataFrame
            training_batch = TrainingDataFrame(training.df[batch_start_index:batch_end_index])

            try:
                # get additional media details
                updated_rows = []

                for idx, row in enumerate(training_batch.df.iter_rows(named=True)):
                    updated_row = collect_details(row)
                    updated_rows.append(updated_row)

                training_batch.update(pl.DataFrame(updated_rows))

                # get media rating metadata
                updated_rows = []

                for idx, row in enumerate(training_batch.df.iter_rows(named=True)):
                    updated_row = collect_ratings(row)
                    updated_rows.append(updated_row)

                training_batch.update(pl.DataFrame(updated_rows))

                # log successfully processed items
                for idx, row in enumerate(training_batch.df.iter_rows(named=True)):
                    logging.info(f"metadata collected - {row['imdb_id']}")

                logging.debug(f"completed metadata collection batch {batch+1}/{number_of_batches}")

            except Exception as e:
                logging.error(f"metadata collection batch {batch+1}/{number_of_batches} failed - {e}")

            try:
                # attempt to write metadata back to the database
                update_training_batch(conn, training_batch)
            except Exception as e:
                logging.error(f"metadata collection batch {batch+1}/{number_of_batches} database update failed - {e}")

            time.sleep(1)

    finally:
        conn.close()


if __name__ == "__main__":
    env = sys.argv[1] if len(sys.argv) > 1 else 'dev'
    collect_metadata(env)