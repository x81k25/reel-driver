# standard library imports
import os

# third-party imports
from dotenv import load_dotenv
import psycopg2

# custom/internal imports
from data_models.training_data_frame import TrainingDataFrame

def extract_media():

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

	con = psycopg2.connect(**con_params)
	cursor = con.cursor()

	with con.cursor() as cursor:
		# Execute the query
		cursor.execute("SELECT * FROM atp.training ORDER BY updated_at")

		# Get column names
		columns = [desc[0] for desc in cursor.description]

		# Fetch all rows
		rows = cursor.fetchall()

		# Convert to dict for polars
		data = [dict(zip(columns, row)) for row in rows]

	# Convert to polars DataFrame and wrap in MediaDataFrame
	training = TrainingDataFrame(data)

	# Close the cursor and connection
	cursor.close()
	con.close()

	# save raw media data
	training.df.write_parquet('./data/01_training.parquet')

# ------------------------------------------------------------------------------
# end of extract_media_table.py
# ------------------------------------------------------------------------------