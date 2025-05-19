# stand libray imports
import json

# third-party imports
import polars as pl

def xgb_prep():
	# ------------------------------------------------------------------------------
	# transform data for xgboost ingestion
	# ------------------------------------------------------------------------------

	# read in data
	training = pl.read_parquet("./data/01_training.parquet")

	# filter to only applicable data
	df = (
		training.filter(pl.col('media_type') == 'movie')
		.select([
			'imdb_id',
			'label',
			'media_title',
			'release_year',
			'genre',
			'language',
			'metascore',
			'rt_score',
			'imdb_rating',
			'imdb_votes'
		])
	)

	# save normalization data for use with model predictions
	normalization = {
		"metascore": {
			"min": df.select(pl.col('metascore').drop_nulls().min())[0, 0],
			"max": df.select(pl.col('metascore').drop_nulls().max())[0, 0]
		},
		"rt_score": {
			"min": df.select(pl.col('rt_score').drop_nulls().min())[0, 0],
			"max": df.select(pl.col('rt_score').drop_nulls().max())[0, 0]
		},
		"imdb_rating": {
			"min": df.select(pl.col('imdb_rating').drop_nulls().min())[0, 0],
			"max": df.select(pl.col('imdb_rating').drop_nulls().max())[0, 0]
		},
		"imdb_votes": {
			"min": df.select(pl.col('imdb_votes').drop_nulls().min())[0, 0],
			"max": df.select(pl.col('imdb_votes').drop_nulls().max())[0, 0]
		},
		"release_year": {
			"min": df.select(pl.col('release_year').drop_nulls().min())[0, 0],
			"max": df.select(pl.col('release_year').drop_nulls().max())[0, 0]
		}
	}

	with open('./model_artifacts/normalization.json', 'w', encoding='utf-8') as file:
		json.dump(normalization, file, indent=2)

	# normalize numeric fields
	df = df.with_columns(
		metascore_norm = ((pl.col('metascore') - pl.col('metascore').min()) /
		 (pl.col('metascore').max() - pl.col('metascore').min())),
		rt_score_norm = ((pl.col('rt_score') - pl.col('rt_score').min()) /
		 (pl.col('rt_score').max() - pl.col('rt_score').min())),
		imdb_rating_norm = ((pl.col('imdb_rating') - pl.col('imdb_rating').min()) /
		 (pl.col('imdb_rating').max() - pl.col('imdb_rating').min())),
		imdb_votes_norm = ((pl.col('imdb_votes') - pl.col('imdb_votes').min()) /
		 (pl.col('imdb_votes').max() - pl.col('imdb_votes').min())),
		release_year_norm = ((pl.col('release_year') - pl.col('release_year').min()) /
		 (pl.col('release_year').max() - pl.col('release_year').min()))
	)

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
	df.write_parquet('./data/02_binomial_classifier_training_data.parquet')

	# ------------------------------------------------------------------------------
	# end of etl.py
	# ------------------------------------------------------------------------------