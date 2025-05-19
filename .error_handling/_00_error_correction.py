# standard library imports
import os

# third-party imports
from dotenv import load_dotenv
import psycopg2

# ------------------------------------------------------------------------------
# setup and params
# ------------------------------------------------------------------------------

load_dotenv()

# create sql connection
con_params = {
	'dbname': os.getenv('PG_DB'),
	'user': os.getenv('PG_USER'),
	'password': os.getenv('PG_PASS'),
	'host': os.getenv('PG_HOST'),
	'port': os.getenv('PG_PORT'),
}

table = "atp.media"

# ------------------------------------------------------------------------------
# support functions
# ------------------------------------------------------------------------------

def get_db_connection():
	"""
	Create and return a PostgreSQL database connection using predefined parameters.

	Returns:
	--------
	psycopg2.connection
		A connection object to the PostgreSQL database
	"""
	try:
		# Create connection using parameters from environment variables
		conn = psycopg2.connect(**con_params)
		return conn
	except Exception as e:
		print(f"Error connecting to database: {e}")
		raise


def update_media_override_status(hashes):
	"""
	Update media records to override rejection status and mark as ingested.

	Parameters:
	-----------
	hashes : list
		List of hash strings to update

	Returns:
	--------
	int
		Number of rows updated
	"""
	# Get a database connection
	conn = get_db_connection()
	cursor = None

	try:
		cursor = conn.cursor()

		# Convert list of hashes to a string format for SQL IN clause
		hash_list = "', '".join(hashes)

		# SQL query with parameterized values
		sql = f"""
        UPDATE {table}
        SET pipeline_status = 'ingested',
            error_status = False,
            error_condition = NULL,
            rejection_status = 'override',
            rejection_reason = NULL
        WHERE hash IN ('{hash_list}')
        """

		# Execute the query
		cursor.execute(sql)
		rows_updated = cursor.rowcount

		# Commit the transaction
		conn.commit()

		return rows_updated

	except Exception as e:
		# Rollback in case of error
		conn.rollback()
		print(f"Error updating media records: {e}")
		raise e

	finally:
		# Close cursor if it was created
		if cursor:
			cursor.close()
		# Always close the connection
		conn.close()

# ------------------------------------------------------------------------------
# execution
# ------------------------------------------------------------------------------

hashes_to_update = [
"59270dee22d3b4d568e87de76c18c5b19507b838",
"761185c0724de8db4362941571ea2c1e16ea950b",
"056000683e522aab55bf8bfee3696ae6f92a5426",
"8679d25ab5b5bea4239a05cd40bef27c49c4e453",
"53652e8a76782c71b4648caa6261049907fc5577",
"43348498d7d945c19bdb4dbde11f97abbd09af9e",
"dd1e9cb242b2d104d48bf52887c75753082d457c",
"61b0855a9dd1c9a4f402810a5f609c070ef19a11",
"42ccd4ae3e2e54c0ad27c4511782822a36c194b1",
"94f871820209b9e6270eb88a91f087f4e0746b66",
"3020cbdb2ddadea1685bc77781ff3289a354e6d2",
"79787a53f25d6238cc6ace856f9480386bf6860d",
"6546a1b91ac070553077d256918305f82cfd232e",
"b44a09d1ad18ceffca1ee9b949a715f28c6b7cf3",
"d0e9f62da63f9b09e845efc2038c22742c35a9f9",
"db46260e97f0b91bf45a53939d84fd7ad909af52",
"f3b0203bff95f4a26a2982e515522382c20db41c",
"f9ed5f53af8311040473bf10b1e52b552ee03d9d",
"8af7cb67d83b057e765449aff28f12d97f12a92e",
"33b35433a86bb7a46f26ae8b8d06d2bdbf6d2e51",
"43711f859699e7cd529a1c9ff67b2ffea1172605",
"3389dc2405b325d46c0824f724594980f21f1620",
"209f1161578d097d52df5084652b7e3e7ebe0beb",
"8bd8ed53ad8be856a7e88b559e06540cd212bb13",
"eb0711bc201c5f7ffce4b7176c1b980323b59f27",
"0a543df47078a88c2d0102f893824cc791971f20",
"3b4d8d09764844d1e481ca99e41b45c0131a234e",
"751a18336a0015268e9bcc1b5845b35819cbe94c",
"d923f4ccc6252acdc1c82ad3933420c4d518d50a",
"b2e6d948d990c1cc901b044ac48958d8b7d10acf",
"f65c89afd5fb228ec1e60e8d1f25838ac25e28d2",
"7c98795f9ffa051d649eb8862677b37eb7198ae9",
"92bf64cda3e5458d257893e3c32766e9343ff2bc",
"34b934694b6da0abe0d26cab9b4a1a7dc415ea4d",
"ec42123f73d1d9f3b93cc5b77c631c931aef7297",
"87aebc3dcb49df19c76a3b7708f5ae8803b2e5d5",
"77e3f8d1993ebc56cf67c084c3b767158f872b58",
"773e1b8129e2ec3b36aa79498588d10181aadb87",
"6c199637677e25561124aa038a6c79ce5271100d",
"dd49310ebe8673b0d7a1f3a9168c91d025ae8000",
"377017e5dff31e591f5cd22329b11ce9696295da",
"3eee734a104ad54b1aa8b998cdf17468abbd6fde",
"611b8895d4f2f4dec674f853193d32be7deddef9",
"95fcf25ac67c22ae31a82ca18a0ba8ee630cd222",
"fc6ac4ee0cbe7bfee330b678feab903ed2bd1c42",
"e5cc5ecb48ae7dbb46a32d84809e7ff1c3df8996",
"e5fa2130f81a3de526a0c427805cbca12e99770f"
]

update_media_override_status(hashes_to_update)

# ------------------------------------------------------------------------------
# end of _00_error_correction.py
# ------------------------------------------------------------------------------