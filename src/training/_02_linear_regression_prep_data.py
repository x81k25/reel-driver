# third-party imports
import polars as pl

# custom/local imports
from src.data_models import MediaDataFrame

# ------------------------------------------------------------------------------
# read in data
# ------------------------------------------------------------------------------

# read in data as MediaDataFrame to standardize
media = MediaDataFrame(pl.read_parquet("./data/media.parquet"))

# filter for movies
df = media.df.filter(pl.col('media_type') == 'movie')

# select only relevant fields
df = df.select([
	'hash',
	'media_title',
	'rejection_status',
	'genre',
	'language',
	'release_year',
	'metascore',
	'rt_score',
	'imdb_rating',
	'imdb_votes'
])

pl.Config.set_tbl_width_chars(1000)
pl.Config.set_tbl_cols(-1)
df.head()

# ------------------------------------------------------------------------------
# set label value
# ------------------------------------------------------------------------------

# convert the rejections status to a continuous value
#   the numeric distinctions here between override and accepted are somewhat
#   arbitrary and should be tweaked and tested for model impact
df = df.with_columns(
	label = pl.when(pl.col('rejection_status') == 'override')
		.then(pl.lit(1))
		.when(pl.col('rejection_status') == 'accepted')
		.then(pl.lit(0.7))
		.otherwise(pl.lit(0))
)

# ------------------------------------------------------------------------------
# sort for unique media objects
# ------------------------------------------------------------------------------

# sort for distinct values, if multiple values for label are present select
#  the highest value
df = (df
	.sort(by=['label', 'media_title'], descending=[True, False])
	.unique(subset=['media_title'])
	.sort(by=['label', 'media_title'], descending=[True, False])
)

# verify
df.group_by('label').agg(pl.len())
df.select(['hash', 'media_title', 'label'])

# ------------------------------------------------------------------------------
# normalize numeric fields
# ------------------------------------------------------------------------------

df = df.with_columns(
	metascore_norm=((pl.col('metascore') - pl.col('metascore').min()) /
	    (pl.col('metascore').max() - pl.col('metascore').min())),
	rt_score_norm=((pl.col('rt_score') - pl.col('rt_score').min()) /
	    (pl.col('rt_score').max() - pl.col('rt_score').min())),
	imdb_rating_norm=((pl.col('imdb_rating') - pl.col('imdb_rating').min()) /
		(pl.col('imdb_rating').max() - pl.col('imdb_rating').min())),
	imdb_votes_norm=((pl.col('imdb_votes') - pl.col('imdb_votes').min()) /
		(pl.col('imdb_votes').max() - pl.col('imdb_votes').min())),
	release_year_norm=((pl.col('release_year') - pl.col('release_year').min()) /
		(pl.col('release_year').max() - pl.col('release_year').min()))
)

# ------------------------------------------------------------------------------
# encode categoricals
# ------------------------------------------------------------------------------

# encode language
# convert to english_language boolean
df = df.with_columns(
	english_language = pl.when(pl.col('language').list.contains("English"))
		.then(pl.lit(1))
		.otherwise(pl.lit(0))
)

# verify
df.select(['language', 'english_language'])

# encode categorical variables
# First explode the genre list as you did before
genre_exploded = df.select(['media_title', 'genre']).explode('genre')

# Count frequency of each genre and get the top 9, excluding nulls
genre_counts = genre_exploded.filter(pl.col('genre').is_not_null()).group_by('genre').len().sort('len', descending=True)
top_9_genres = genre_counts.head(9).select('genre').to_series().to_list()

# Create columns for top 9 genres
for genre in top_9_genres:
    df = df.with_columns(
        pl.when(pl.col('genre').is_null())
        .then(0)  # Set to 0 for null genre items
        .otherwise(
            # Check if this specific genre is in the list
            pl.col('genre').list.contains(genre).cast(pl.Int8)
        )
        .alias(f"genre_{genre}")
    )

# Create a helper column by summing up all genre columns
# If sum > 0, the item has at least one top genre
genre_columns = [pl.col(f"genre_{genre}") for genre in top_9_genres]
df = df.with_columns(
    pl.sum_horizontal(genre_columns).alias("has_top_genre")
)

# Now create the other column based on the helper column
df = df.with_columns(
    pl.when(pl.col('genre').is_null())
    .then(0)
    .otherwise(
        pl.when((pl.col('has_top_genre') == 0) & (pl.col('genre').list.len() > 0))
        .then(1)
        .otherwise(0)
    ).cast(pl.Int8).alias("genre_other")
)

# ------------------------------------------------------------------------------
# clean data for export and export
# ------------------------------------------------------------------------------

# fill all remaing nulls with 0
df = df.fill_null(0)

# select only desired fields
df = df.select([
	'hash',
	'media_title',
	'label',
	'metascore_norm',
	'rt_score_norm',
	'imdb_rating_norm',
	'imdb_votes_norm',
	'release_year_norm',
	'english_language',
	pl.col('genre_Drama').alias('genre_drama'),
	pl.col('genre_Comedy').alias('genre_comedy'),
	pl.col('genre_Thriller').alias('genre_thriller'),
	pl.col('genre_Action').alias('genre_action'),
	pl.col('genre_Documentary').alias('genre_documentary'),
	pl.col('genre_Horror').alias('genre_horror'),
	pl.col('genre_Romance').alias('genre_romance'),
	pl.col('genre_Crime').alias('genre_crime'),
	pl.col('genre_Adventure').alias('genre_adventure'),
	'genre_other'
])

# verify normalized values
print(df.select([
	'metascore_norm',
	'rt_score_norm',
	'imdb_rating_norm',
	'imdb_votes_norm',
	'release_year_norm'
]).describe())

# verify encoded values
print(df.select([
	'genre_drama',
	'genre_comedy',
	'genre_thriller',
	'genre_action',
	'genre_documentary'
]).describe())

print(df.select([
	'genre_horror',
	'genre_romance',
	'genre_crime',
	'genre_adventure',
	'genre_other'
]).describe())

print(df.select([
	'hash',
	'media_title',
	'label',
	'english_language'
]).describe())

df.head()

# write data file
df.write_parquet('./data/linear_regression_training_data.parquet')

# ------------------------------------------------------------------------------
# end of linear_regression_prep_data.py
# ------------------------------------------------------------------------------