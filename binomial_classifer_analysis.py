# third-party imports
import pandas as pd
import polars as pl

# ------------------------------------------------------------------------------
# create model analysis table
# ------------------------------------------------------------------------------

# read in results
results_df = pl.read_parquet('./data/binomial_classifier_results.parquet')

# read in raw media data
media = pl.read_parquet('./data/media.parquet')

# join datset
pl_df = results_df.join(
	media,
	on='hash',
	how='left'
)

# store predicted as boolean
pl_df = pl_df.with_columns(
	predicted = pl.col('predicted').cast(pl.Boolean)
)

# crate column for confusion category
pl_df = pl_df.with_columns(
	conf_cat = pl.when((pl.col('actual') == True) & (pl.col('predicted') == True))
		.then(pl.lit("tp"))
		.when((pl.col('actual') == True) & (pl.col('predicted') == False))
		.then(pl.lit("fn"))
		.when((pl.col('actual') == False) & (pl.col('predicted') == False))
		.then(pl.lit("tn"))
		.when((pl.col('actual') == False) & (pl.col('predicted') == True))
		.then(pl.lit("fp"))
		.otherwise(None)
)

# sort by confusion category
pl_df = pl_df.sort('conf_cat')

# re-order df
pl_df = pl_df.select([
	'hash',
	'media_title',
	'actual',
	'predicted',
	'conf_cat',
	'probability',
	'release_year',
	'rt_score',
	'metascore',
	'imdb_rating',
	'imdb_votes',
	'genre',
	'language'
])

# ------------------------------------------------------------------------------
# analyze model results
# ------------------------------------------------------------------------------

# convert df to pandas
df = pl_df.to_pandas()




# ------------------------------------------------------------------------------
# end of binomial_classifier_results.py
# ------------------------------------------------------------------------------