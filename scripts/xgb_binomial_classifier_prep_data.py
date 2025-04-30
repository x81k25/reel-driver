# third-party imports
import polars as pl

# ------------------------------------------------------------------------------
# transform data for xgboost ingestion
# ------------------------------------------------------------------------------

# read in data
media = pl.read_parquet("./data/media.parquet")

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
df = (df
	.sort(by=["label", "media_title"], descending=[True, False])
	.unique(subset=['media_title'])
)

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