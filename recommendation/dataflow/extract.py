import os
import argparse
import logging
from datetime import datetime

import apache_beam as beam
from apache_beam.options.pipeline_options import \
    PipelineOptions

PROJECT = 'qwiklabs-gcp-ml-2c89a2800f38'


def create_query(phase):
    base_query = """
    SELECT
        *,
        MOD(ABS(FARM_FINGERPRINT(CAST(timestamp AS STRING))), 10) AS hash_value
    FROM
        `qwiklabs-gcp-ml-2c89a2800f38.dev_recommendation.ratings`
    """

    if phase == 'TRAIN':
        subsumple = """
        hash_value < 7
        """
    elif phase == 'TEST':
        subsumple = """
        hash_value >= 7
        """

    query = """
    SELECT 
        userId, 
        movieId, 
        rating 
    FROM 
        ({0})
    WHERE {1}
    """.\
        format(base_query, subsumple)

    return query


def to_csv(line):
    csv_columns = 'userId,movieId,rating'.split(',')
    rowstring = ','.join([str(line[k]) for k in csv_columns])
    return rowstring


def get_date():
    jst_now = datetime.now()
    dt = datetime.strftime(jst_now, "%Y-%m-%d")

    return dt


def run(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output',
        required=True
    )

    known_args, pipeline_args = \
        parser.parse_known_args(argv)

    options = PipelineOptions(pipeline_args)
    with beam.Pipeline(options=options) as p:
        for phase in ['TRAIN', 'TEST']:
            query = create_query(phase)

            date = get_date()
            output_path = os.path.join(known_args.output, date,
                                       "{}.csv".format(phase))

            read = p | 'ExtractFromBigQuery_{}'.format(phase) >> beam.io.Read(
                beam.io.BigQuerySource(
                    project=PROJECT,
                    query=query,
                    use_standard_sql=True
                )
            )

            convert = read | 'ConvertToCSV_{}'.format(phase) >> beam.Map(to_csv)

            convert | 'WriteToGCS_{}'.format(phase) >> beam.io.Write(
                beam.io.WriteToText(output_path))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
