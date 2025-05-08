# standard library imports
import concurrent.futures
import os

# third-party imports
from dotenv import load_dotenv
import polars as pl
import psycopg2

def extract_media():

	# ------------------------------------------------------------------------------
	# setup and params
	# ------------------------------------------------------------------------------

	load_dotenv()

	# ------------------------------------------------------------------------------
	# extract data from database
	# ------------------------------------------------------------------------------

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

	# execute query
	cursor.execute("SELECT * FROM atp.media order by updated_at")
	results = cursor.fetchall()

	# convert to pl.df
	schema = {
		'hash': pl.Utf8,
		'media_type': pl.Categorical,
		'media_title': pl.Utf8,
		'season': pl.Int64,
		'episode': pl.Int64,
		'release_year': pl.Int64,
		'pipeline_status': pl.Categorical,
		'error_status': pl.Boolean,
		'error_condition': pl.Utf8,
		'rejection_status': pl.Categorical,
		'rejection_reason': pl.Utf8,
		'parent_path': pl.Utf8,
		'target_path': pl.Utf8,
		'original_title': pl.Utf8,
		'original_path': pl.Utf8,
		'original_link': pl.Utf8,
		'rss_source': pl.Categorical,
		'uploader': pl.Utf8,
		'genre': pl.List(pl.Utf8),
		'language': pl.List(pl.Utf8),
		'rt_score': pl.Int64,
		'metascore': pl.Int64,
		'imdb_rating': pl.Float64,
		'imdb_votes': pl.Int64,
		'imdb_id': pl.Utf8,
		'resolution': pl.Utf8,
		'video_codec': pl.Utf8,
		'upload_type': pl.Utf8,
		'audio_codec': pl.Utf8,
		'created_at': pl.Datetime,
		'updated_at': pl.Datetime,
		'tmdb_id': pl.Int64
	}

	media = pl.DataFrame(schema=schema)

	def parallel_process(results, media):
		def process_row(row):
			try:
				return pl.DataFrame([row], orient='row', schema=media.schema)
			except Exception as e:
				print(f"could not load {row} - error: {e}")
				return None

		valid_dfs = []

		with concurrent.futures.ThreadPoolExecutor() as executor:
			future_to_row = {executor.submit(process_row, row): row for row in
							 results}
			for future in concurrent.futures.as_completed(future_to_row):
				df = future.result()
				if df is not None:
					valid_dfs.append(df)

		# Concatenate collected DataFrames if any exist
		if valid_dfs:
			# Create a new DataFrame by concatenating all valid DataFrames
			new_rows = pl.concat(valid_dfs)
			# Then stack the new rows with the original media DataFrame
			return pl.concat([media, new_rows])
		return media

	media = parallel_process(results, media)

	# Close the cursor and connection
	cursor.close()
	con.close()

	# save raw media data
	media.write_parquet('./data/media.parquet')

	# ------------------------------------------------------------------------------
	# end of extract_media_table.py
	# ------------------------------------------------------------------------------