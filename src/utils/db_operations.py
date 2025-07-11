# standard library imports
import os

# third-party imports
from dotenv import load_dotenv
from loguru import logger
import polars as pl
import psycopg2
import psycopg2.extras

# ------------------------------------------------------------------------------
# setup and config
# ------------------------------------------------------------------------------

# load dotenv at the module level if running locally
if os.getenv("LOCAL_DEVELOPMENT", '') == "true":
    load_dotenv()

# ------------------------------------------------------------------------------
# supporting functions
# ------------------------------------------------------------------------------

def gen_pg2_con(
    host: str = os.getenv('REEL_DRIVER_TRNG_PGSQL_HOST'),
    port: str = int(os.getenv('REEL_DRIVER_TRNG_PGSQL_PORT', 5432)),
    dbname: str = os.getenv('REEL_DRIVER_TRNG_PGSQL_DATABASE'),
    schema: str = os.getenv('REEL_DRIVER_TRNG_PGSQL_SCHEMA'),
    user: str = os.getenv('REEL_DRIVER_TRNG_PGSQL_USERNAME'),
    password: str = os.getenv('REEL_DRIVER_TRNG_PGSQL_PASSWORD')
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
    schema: str = os.getenv("REEL_DRIVER_TRNG_PGSQL_SCHEMA")
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
# insert functions
# ------------------------------------------------------------------------------

def trunc_and_load(
        df: pl.DataFrame,
        table_name: str,
        schema: str = os.getenv("REEL_DRIVER_TRNG_PGSQL_SCHEMA"),
        truncate: bool = True
    ):
    """
    Insert DataFrame rows into specified table.

    :param df: polars dataframe to be loaded
    :param table_name: name of table in the database to load to
    :param schema: database to schema to which table belongs
    :param truncate: whether or not to truncate the current able before
        inserting the new value
    :return: None
    """
    logger.info(f"inserting {table_name} to db")

    con = gen_pg2_con()
    with con.cursor() as cur:
        if truncate:
            cur.execute(f"TRUNCATE TABLE {schema}.{table_name};")

        # Auto-generate column list and placeholders
        columns = ", ".join(df.columns)
        data = [tuple(row) for row in df.iter_rows()]

        psycopg2.extras.execute_values(
            cur,
            f"INSERT INTO {schema}.{table_name} ({columns}) VALUES %s",
            data,
            template=None,
            page_size=1000
        )

        con.commit()

    logger.info(f"{table_name} loaded")


# ------------------------------------------------------------------------------
# end of db_operations.py
# ------------------------------------------------------------------------------