# third-party imports
from loguru import logger
import polars as pl
import re

# custom/local imports
import src.utils as utils

# -----------------------------------------------------------------------------
# supporting functions
# -----------------------------------------------------------------------------

def one_hot_encode_list_column(
    series: pl.Series,
    column_name: str
) -> tuple[list[str], list[list[int]]]:
    """
	One-hot encode a Polars Series containing lists of strings.

	:param series: pl.Series containing lists of strings to be one-hot encoded
	:param column_name: str, original column name to use as prefix for generated column names
	:return: tuple containing list of column names and list of one-hot encoded arrays
	"""

    def clean_value(value: str) -> str:
        # Replace non-word characters with underscores and convert to lowercase
        cleaned = re.sub(r'[^\w]', '_', value.lower())
        # Replace multiple consecutive underscores with single underscore
        cleaned = re.sub(r'_+', '_', cleaned)
        # Remove leading/trailing underscores
        return cleaned.strip('_')

    # Handle null/empty cases
    if series.is_null().all():
        return [column_name], [[0] for _ in range(len(series))]

    # Get all unique values across all lists
    all_values = set()
    for row in series:
        if row is not None and len(row) > 0:
            all_values.update(row)

    # If no values found, return original column name
    if not all_values:
        return [column_name], [[0] for _ in range(len(series))]

    # Create sorted list of unique values and their cleaned column names
    sorted_values = sorted(all_values)
    column_names = [f"{column_name}_{clean_value(value)}" for value in
                    sorted_values]

    # Create one-hot encoded arrays
    encoded_arrays = []
    for row in series:
        if row is None or len(row) == 0:
            encoded_arrays.append([0] * len(sorted_values))
        else:
            encoded_row = [1 if value in row else 0 for value in sorted_values]
            encoded_arrays.append(encoded_row)

    return column_names, encoded_arrays


# -----------------------------------------------------------------------------
# transform
# -----------------------------------------------------------------------------

def filter_training(training: pl.DataFrame) -> pl.DataFrame:
    """
    get the full training data table from pgsql

    :param training: input DataFrame
    :return: output DataFrame with transformation complete
    """
    training_filtered = training.clone()

    logger.info(f"performing filtering on {training_filtered.height} row of training data")

    training_filtered = (
        training_filtered
            # we are currently only analyzing move data
            .filter(pl.col('media_type') == 'movie')
            # pull in only training data that has been confirmed for accuracy
            .filter(pl.col('reviewed'))
            # select only field which will actually be used
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

    logger.info(f"returning {training_filtered.height} rows filtered data")

    return training_filtered


def num_training(training: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    perform transformation on all numeric fields

    :param training: input training DataFrame
    :return: tuple containing [
        DataFrame with properly processed numeric fields,
        DataFrame of the range of values to be used for future normalizations
    ]
    """
    logger.info("performing basic numerical transformation on training")

    training_num = training.clone()

    num_columns = [
        "release_year",
        "budget",
        "revenue",
        "runtime",
        "tmdb_rating",
        "tmdb_votes",
        "rt_score",
        "metascore",
        "imdb_rating",
        "imdb_votes"
    ]

    # create dataframe to hold min_max values
    # instantiate DataFrame to hold column_name_mappings
    normalization_table = pl.DataFrame(schema={
        "feature": pl.String,
        "min": pl.Float64,
        "max": pl.Float64
    })

    for num_column in num_columns:
        # get normalized values and add to normalization_table
        col_min = float(training_num.select(pl.col(num_column).drop_nulls().min())[0, 0])
        col_max = float(training_num.select(pl.col(num_column).drop_nulls().max())[0, 0])

        schema_row = pl.DataFrame({
            'feature': num_column,
            'min': col_min,
            'max': col_max
        })
        normalization_table = pl.concat([normalization_table, schema_row])

        # get normalized values
        training_num = training_num.with_columns(
            (pl.col(num_column) - pl.lit(col_min)) /
            (pl.lit(col_max) - pl.lit(col_min))
            .alias(num_column)
        )

    logger.info("basic numerical transformation complete")

    return training_num, normalization_table


def cat_training(training: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    perform transformation on all categorical fields

    :param training: input training DataFrame
    :return: DataFrame with properly processed categorical fields
    :return: DataFrame containing the column_name mappings for sparse matrices
        which will be stored as vectors inside of the primary DataFrame
    """
    logger.info("performing categorical operations on training data")

    training_cat = training.clone()

    # instantiate DataFrame to hold column_name_mappings
    engineered_schema = pl.DataFrame(schema={
        "original_column": pl.String,
        "exploded_mapping": pl.List(pl.String)
    })

    # production_companies currently not being loaded as it would balloon total data_set size
    training_cat = training_cat.drop('production_companies')

    list_columns = [
        'origin_country',
        'production_countries',
        'spoken_languages',
        'genre'
    ]

    # for each categorical fields
    #   explode column and create sparse matrices
    #   store column mappings in schema DataFrame
    #   store column value as an array in the original column of the data frame
    for list_column in list_columns:
        # encode values
        names, values = one_hot_encode_list_column(
            series=training_cat[list_column],
            column_name=list_column
        )

        # Add as new column
        training_cat = (
                training_cat.with_columns(
                pl.Series(
                    (list_column + '_encoded'), values
                )
            ).drop(list_column)
            .rename({(list_column + '_encoded'): list_column})
        )

        schema_row = pl.DataFrame({
            'original_column': [list_column],
            'exploded_mapping': [names]
        })
        engineered_schema = pl.concat([engineered_schema, schema_row])

    logger.info("categorical operations complete")

    return training_cat, engineered_schema


def encode_training(training: pl.DataFrame) -> pl.DataFrame:
    """
    encoded training label for model ingestion based on relevant featurs

    :param training: input training DataFrame
    :return: DataFrame with training label encoded
    """
    logger.info("encoding training label")

    training_encoded = training.clone()

    training_encoded = training_encoded.with_columns(
        label=pl.when(pl.col('label') == 'would_watch')
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
    )

    logger.info("training label encoded")

    return training_encoded


# -----------------------------------------------------------------------------
# primary function
# -----------------------------------------------------------------------------

def xgb_prep():
    """
    full pipeline to transform training data into a fully engineered feature
        set ready for ingestion and storage back into the database
    """

    # get full data set
    training = utils.select_star(table="training")

    # perform preliminary polars operations
    engineered = filter_training(training)

    # perform numeric transformation
    engineered, normalization_table = num_training(engineered)

    # perform categorical transformations
    engineered, engineered_schema = cat_training(engineered)

    # encode training label
    engineered = encode_training(engineered)

    # print summary information
    logger.info(engineered.group_by('label').agg(pl.len()))
    logger.info(engineered.select([
        'budget',
        'revenue',
        'runtime',
        'release_year',
        'tmdb_rating',
        'tmdb_votes',
        'rt_score',
        'metascore',
        'imdb_rating',
        'imdb_votes'
    ]).describe())

    # store all engineered training values and metadata
    utils.trunc_and_load(
        df=engineered,
        table_name="engineered"
    )
    utils.trunc_and_load(
        df=normalization_table,
        table_name="engineered_normalization_table"
    )
    utils.trunc_and_load(
        df=engineered_schema,
        table_name="engineered_schema"
    )


# main guard
def __main__():
    xgb_prep()

# ------------------------------------------------------------------------------
# end of _02_xgb_binomial_classifier_prep_data.py
# ------------------------------------------------------------------------------