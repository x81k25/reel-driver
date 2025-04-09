# standard library imports
import concurrent.futures
import os

# third-party imports
from dotenv import load_dotenv
import polars as pl
import psycopg2

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
    'updated_at': pl.Datetime
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
# transform data for xgboost ingestion
# ------------------------------------------------------------------------------

# filter by medias only for now
df = media.filter(pl.col('media_type') == 'movie')

# select only relevant fields
df = df.select([
	'hash',
	'media_title',
	'rejection_status',
	'release_year',
	'genre',
	'language',
	'metascore',
	'rt_score',
	'imdb_rating',
	'imdb_votes'
])

# convert rejection status to label
df = df.with_columns(
	label = pl.when(pl.col('rejection_status') == 'accepted')
		.then(True)
		.when(pl.col('rejection_status') == 'override')
		.then(True)
		.otherwise(False)
	)

# label counts
label_counts = df.group_by('label').agg(pl.len())
print(label_counts)

# get count of how many times the same media titles shows up in that data
results = df.group_by('media_title').agg(
	pl.len().alias('len')).sort('len', descending=True)

print(results)

# get only distinct media titles
## sort the data so that values of 1 show up first
df = df.sort(by=["label", "media_title"], descending=[True, False])

## grab unique values starting from the top
df = df.unique(subset=['media_title'])

# ensure all positive labels are preserved: ~300
label_counts = df.group_by('label').agg(pl.len())
print(label_counts)

# normalize numeric fields
df = df.with_columns([
	((pl.col('metascore') - pl.col('metascore').min()) /
	    (pl.col('metascore').max() - pl.col('metascore').min()))
		.alias('metascore_norm'),
	((pl.col('rt_score') - pl.col('rt_score').min()) /
	    (pl.col('rt_score').max() - pl.col('rt_score').min()))
		.alias('rt_score_norm'),
	((pl.col('imdb_rating') - pl.col('imdb_rating').min()) /
	 	(pl.col('imdb_rating').max() - pl.col('imdb_rating').min()))
		.alias('imdb_rating_norm'),
	((pl.col('imdb_votes') - pl.col('imdb_votes').min()) /
	 	(pl.col('imdb_votes').max() - pl.col('imdb_votes').min()))
		.alias('imdb_votes_norm'),
	((pl.col('release_year') - pl.col('release_year').min()) /
	 	(pl.col('release_year').max() - pl.col('release_year').min()))
		.alias('release_year_norm'),
])

# encode categorical variables
genre_exploded = df.select(['media_title', 'genre']).explode('genre')

unique_genres = genre_exploded.select('genre').unique().to_series().to_list()

for genre in unique_genres:
	df = df.with_columns(
		pl.lit(genre).is_in(pl.col('genre')).cast(pl.Int8).alias(f"genre_{genre}")
	)

## encode langauge
language_exploded = df.select(['media_title', 'language']).explode('language')

unique_languages = language_exploded.select('language').unique().to_series().to_list()

for language in unique_languages:
	df = df.with_columns(
		pl.lit(language).is_in(pl.col('language')).cast(pl.Int8).alias(f"language_{language}")
	)

# select training data only
df = df.drop([
	'rejection_status',
	'release_year',
	'genre',
	'language',
	'metascore',
	'rt_score',
	'imdb_rating',
	'imdb_votes'
])

# label counts
label_counts = df.group_by('label').agg(pl.len())
print(label_counts)

# numerica columns
print(df.select([
	'metascore_norm',
	'rt_score_norm',
	'imdb_rating_norm',
	'imdb_votes_norm',
	'release_year_norm'
]).describe())

# write data file
df.write_parquet('./data/binomial_classifier_training_data.parquet')

# ------------------------------------------------------------------------------
# end of etl.py
# ------------------------------------------------------------------------------