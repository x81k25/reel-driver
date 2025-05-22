CREATE VIEW atp.rda AS
select
    -- identifier fields
    pred.imdb_id,
	train.media_title,
	-- prediction fields
	train.label,
	pred.prediction,
	pred.probability,
	pred.cm_value,
	-- flag fields
	train.human_labeled,
	train.anomalous,
	-- training features
	-- - time metadata
	train.release_year,
	-- - quantitative fields
    train.budget,
    train.revenue,
    train.runtime,
    -- - country and production information
    train.origin_country,
    train.production_companies,
    train.production_countries,
    train.production_status,
    -- - language information
    train.original_language,
    train.spoken_languages,
    -- - other string fields
    train.genre,
    train.original_media_title,
    -- - long string fields
    train.tagline,
    train.overview,
    -- - ratings info
    train.tmdb_rating,
    train.tmdb_votes,
    train.rt_score,
    train.metascore,
    train.imdb_rating,
    train.imdb_votes
from atp.prediction as pred
inner join atp.training as train on pred.imdb_id = train.imdb_id;