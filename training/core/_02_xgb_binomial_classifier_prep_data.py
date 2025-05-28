# stand libray imports
import json

# third-party imports
import polars as pl
import re

# -----------------------------------------------------------------------------
# supporting functions
# -----------------------------------------------------------------------------

def encode_list_column(
    column_data: pl.Series,
    column_name: str
) -> pl.DataFrame:
    """
    Convert a column of string lists into binary indicator columns.

    :param column_data: pl.Series containing lists of strings
    :param column_name: str, name to use for column prefixes
    :return: pl.DataFrame with binary columns for each unique value
    """
    # Create temporary df for exploding
    temp_df = pl.DataFrame({column_name: column_data})
    exploded = temp_df.explode(column_name)

    # Get unique values, filtering out None/null values
    raw_unique_values = exploded.select(column_name).drop_nulls().unique().to_series().to_list()

    # Group values by their cleaned names
    clean_name_groups = {}
    for value in raw_unique_values:
        if value is not None:
            clean_value = re.sub(r'[^\w]', '_', value.lower())
            if clean_value not in clean_name_groups:
                clean_name_groups[clean_value] = []
            clean_name_groups[clean_value].append(value)

    # Create binary columns
    result_columns = []
    for clean_name, original_values in clean_name_groups.items():
        # Check if ANY of the original values exist in each row
        binary_col = pl.any_horizontal([
            pl.lit(val).is_in(column_data) for val in original_values
        ]).cast(pl.Int8).alias(f"{column_name}_{clean_name}")
        result_columns.append(binary_col)

    return pl.DataFrame().with_columns(result_columns)


# -----------------------------------------------------------------------------
# main function
# -----------------------------------------------------------------------------

def xgb_prep():

    # -------------------------------------------------------------------------
    # transform data for xgboost ingestion
    # -------------------------------------------------------------------------

    # read in data
    training = pl.read_parquet("./data/01_training.parquet")

    # filter to only applicable data
    df = (
        training.filter(pl.col('media_type') == 'movie')
        .select([
            # identifiers
            'imdb_id',
            'media_title',
            # label
            'label',
            # continuous
            'release_year',
            'budget',
            'revenue',
            'runtime',
            'tmdb_rating',
            'tmdb_votes',
            'rt_score',
            'metascore',
            'imdb_rating',
            'imdb_votes',
            # categorical
            'origin_country',
            'production_status',
            # categorical lists
            'production_companies',
            'production_countries',
            'original_language',
            'spoken_languages',
            'genre',
        ])
    )

    # encode label
    df = df.with_columns(
        label = pl.when(pl.col('label') == 'would_watch')
            .then(pl.lit(1))
        .otherwise(pl.lit(0))
    )

    # save normalization data for use with model predictions
    normalization = {
        "release_year": {
            "min": df.select(pl.col('release_year').drop_nulls().min())[0, 0],
            "max": df.select(pl.col('release_year').drop_nulls().max())[0, 0]
        },
        "budget": {
            "min": df.select(pl.col('budget').drop_nulls().min())[0, 0],
            "max": df.select(pl.col('budget').drop_nulls().max())[0, 0]
        },
        "revenue": {
            "min": df.select(pl.col('revenue').drop_nulls().min())[0, 0],
            "max": df.select(pl.col('revenue').drop_nulls().max())[0, 0]
        },
        "runtime": {
            "min": df.select(pl.col('runtime').drop_nulls().min())[0, 0],
            "max": df.select(pl.col('runtime').drop_nulls().max())[0, 0]
        },
        "tmdb_rating": {
            "min": df.select(pl.col('tmdb_rating').drop_nulls().min())[0, 0],
            "max": df.select(pl.col('tmdb_rating').drop_nulls().max())[0, 0]
        },
        "tmdb_votes": {
            "min": df.select(pl.col('tmdb_votes').drop_nulls().min())[0, 0],
            "max": df.select(pl.col('tmdb_votes').drop_nulls().max())[0, 0]
        },
        "rt_score": {
            "min": df.select(pl.col('rt_score').drop_nulls().min())[0, 0],
            "max": df.select(pl.col('rt_score').drop_nulls().max())[0, 0]
        },
        "metascore": {
            "min": df.select(pl.col('metascore').drop_nulls().min())[0, 0],
            "max": df.select(pl.col('metascore').drop_nulls().max())[0, 0]
        },
        "imdb_rating": {
            "min": df.select(pl.col('imdb_rating').drop_nulls().min())[0, 0],
            "max": df.select(pl.col('imdb_rating').drop_nulls().max())[0, 0]
        },
        "imdb_votes": {
            "min": df.select(pl.col('imdb_votes').drop_nulls().min())[0, 0],
            "max": df.select(pl.col('imdb_votes').drop_nulls().max())[0, 0]
        }
    }

    with open('./model_artifacts/normalization.json', 'w', encoding='utf-8') as file:
        json.dump(normalization, file, indent=2)

    # normalize numeric fields and drop source columns
    df =df.with_columns(
        release_year_norm = ((pl.col('release_year') - pl.col('release_year').min()) /
            (pl.col('release_year').max() - pl.col('release_year').min())),
        budget_norm = ((pl.col('budget') - pl.col('budget').min()) /
            (pl.col('budget').max() - pl.col('budget').min())),
        revenue_norm = ((pl.col('revenue') - pl.col('revenue').min()) /
            (pl.col('revenue').max() - pl.col('revenue').min())),
        runtime_norm = ((pl.col('runtime') - pl.col('runtime').min()) /
            (pl.col('runtime').max() - pl.col('runtime').min())),
        tmdb_rating_norm = ((pl.col('tmdb_rating') - pl.col('tmdb_rating').min()) /
            (pl.col('tmdb_rating').max() - pl.col('tmdb_rating').min())),
        tmdb_votes_norm = ((pl.col('tmdb_votes') - pl.col('tmdb_votes').min()) /
            (pl.col('tmdb_votes').max() - pl.col('tmdb_votes').min())),
        rt_score_norm = ((pl.col('rt_score') - pl.col('rt_score').min()) /
            (pl.col('rt_score').max() - pl.col('rt_score').min())),
        metascore_norm = ((pl.col('metascore') - pl.col('metascore').min()) /
            (pl.col('metascore').max() - pl.col('metascore').min())),
        imdb_rating_norm = ((pl.col('imdb_rating') - pl.col('imdb_rating').min()) /
            (pl.col('imdb_rating').max() - pl.col('imdb_rating').min())),
        imdb_votes_norm = ((pl.col('imdb_votes') - pl.col('imdb_votes').min()) /
            (pl.col('imdb_votes').max() - pl.col('imdb_votes').min()))
    ).drop([
	  	'release_year',
        'budget',
        'revenue',
        'runtime',
	  	'tmdb_rating',
		'tmdb_votes',
	  	'rt_score',
		'metascore',
        'imdb_rating',
        'imdb_votes'
	])

    # currently not included in training set as it would add 5148 columns
    #production_companies_encoded = encode_list_column(df['production_companies'], 'production_company')
    #df = pl.concat([df, production_companies_encoded], how='horizontal')
    df = df.drop('production_companies')

    origin_country_encoded = encode_list_column(df['origin_country'], 'origin_country')
    df = pl.concat([df, origin_country_encoded], how='horizontal').drop('origin_country')

    production_countries_encoded = encode_list_column(df['production_countries'], 'production_country')
    df = pl.concat([df, production_countries_encoded], how='horizontal').drop('production_countries')

    spoken_languages_encoded = encode_list_column(df['spoken_languages'], 'spoken_language')
    df = pl.concat([df, spoken_languages_encoded], how='horizontal').drop('spoken_languages')

    # encode categorical list columns
    genre_encoded = encode_list_column(df['genre'], 'genre')
    df = pl.concat([df, genre_encoded], how='horizontal').drop('genre')

    # label counts
    label_counts = df.group_by('label').agg(pl.len())
    print(label_counts)

    # numerica columns
    print(df.select([
        'budget_norm',
        'revenue_norm',
        'runtime_norm',
        'release_year_norm',
        'tmdb_rating_norm',
		'tmdb_votes_norm',
        'rt_score_norm',
        'metascore_norm',
        'imdb_rating_norm',
        'imdb_votes_norm'
    ]).describe())

    # write data file
    df.write_parquet('./data/02_binomial_classifier_training_data.parquet')

# ------------------------------------------------------------------------------
# end of _02_xgb_binomial_classifier_prep_data.py
# ------------------------------------------------------------------------------