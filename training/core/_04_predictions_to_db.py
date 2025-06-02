# third-party imports
import polars as pl
import psycopg2
from psycopg2.extensions import connection
import yaml

def predictions_to_db():

    # read in training data
    df = pl.read_parquet('./data/03_binomial_classifier_results.parquet')

    # create confusion matrix field
    df = df.with_columns(
        cm_value = pl.when(pl.col("actual") == 1)
            .then(
                pl.when(pl.col("predicted") == 1)
                    .then(pl.lit("TP"))
                    .otherwise(pl.lit("FN"))
            ).otherwise(
                pl.when(pl.col("predicted") == 1)
                    .then(pl.lit("FP"))
                    .otherwise(pl.lit("TN"))
            )
    )

    df = df.select([
        'imdb_id',
        pl.col('predicted').alias('prediction'),
        'probability',
        'cm_value'
    ])

    # create db conn
    def create_conn(env: str = 'prod') -> psycopg2.extensions.connection:
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

    conn = create_conn()

    # truncate existing table and peform insert
    with conn.cursor() as cursor:
        # truncate existing table
        cursor.execute("TRUNCATE TABLE atp.prediction")

        # create statement to insert new data
        statement = """
        INSERT INTO atp.prediction (imdb_id, prediction, probability, cm_value)
        VALUES (%s, %s, %s, %s)
        """

        # convert Polars DataFrame to list of tuples for psycopg2
        records = df.to_numpy().tolist()
        # execute many to insert all rows at once
        cursor.executemany(statement, records)
        rows_inserted = cursor.rowcount
        conn.commit()

    conn.close()

    print(f"rows inserted into prediction: {rows_inserted}")

# ------------------------------------------------------------------------------
# end of _04_predictions_to_db.py
# ------------------------------------------------------------------------------