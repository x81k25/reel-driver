# standard library imports
import os

# third-party imports
from dotenv import load_dotenv
from loguru import logger
import polars as pl
import psycopg2

# ------------------------------------------------------------------------------
# setup and config
# ------------------------------------------------------------------------------

# load dotenv at the module level if running locally
#if os.getenv('LOCAL_DEVELOPMENT', '').lower() == 'true':
load_dotenv()

# ------------------------------------------------------------------------------
# supporting functions
# ------------------------------------------------------------------------------

def gen_pg2_con(
    host: str = os.getenv('REEL_DRIVER_HOST'),
    port: str = os.getenv('REEL_DRIVER_PORT'),
    dbname: str = os.getenv('REEL_DRIVER_DATABASE'),
    schema: str = os.getenv('REEL_DRIVER_SCHEMA'),
    user: str = os.getenv('REEL_DRIVER_USERNAME'),
    password: str = os.getenv('REEL_DRIVER_PASSWORD')
) -> psycopg2.connect:
    """
    function to create and return a psycopg2 connection object

    :param host: automatic-transmission database host
    :param port: automatic-transmission database port
    :param dbname: automatic-transmission database name
    :param schema: automatic-transmission schema name
    :param user: reel-driver service account username
    :param password: reel-driver service account password
    :return:
    """

    con_params = {
        'dbname': dbname,
        'user': user,
        'password': password,
        'host': host,
        'port': port,
    }

    con = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )

    return con


# ------------------------------------------------------------------------------
# select functions
# ------------------------------------------------------------------------------

def select_star(
    table: str,
    schema: str = "atp"
) -> pl.DataFrame:
    """
    get the full training data table from pgsql

    :param table: string name of table to SELECT * FROM
    :param schema: database scheme to extract from
    :return: DataFrame of all needed training data
    """
    con = gen_pg2_con()

    with con.cursor() as cursor:
        # build query
        query = f"SELECT * FROM {schema}.{table}"

        logger.info(f"executing: {query}")

        # execute query
        cursor.execute(query)

        # Get column names
        columns = [desc[0] for desc in cursor.description]

        # Fetch all rows
        rows = cursor.fetchall()

        # Convert to dict for polars
        data = [dict(zip(columns, row)) for row in rows]

    logger.info(f"retrieved {len(data)} rows from {schema}.{table}")

    df = pl.DataFrame(data)

    con.close()

    return df


# ------------------------------------------------------------------------------
# end of db_operations.py
# ------------------------------------------------------------------------------