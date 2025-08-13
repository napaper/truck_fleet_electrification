"""
Database utility functions for truck fleet electrification analysis.

IMPORTANT NOTE: This script is ONLY required when database access is needed.
The current analysis workflow uses CSV files instead of database queries, so this
script is not essential for the main analyses.

This module provides:
- PostgreSQL database connection management
- SQL query execution with caching
- Geospatial data processing (GeoDataFrames)

Keep this script if you plan to:
- Re-integrate database access in the future
- Work with the original PostgreSQL data source
- Need geospatial data processing capabilities

For CSV-based analysis, this script can be safely ignored.
"""

import pandas as pd
import sqlalchemy
import joblib
import geopandas as gpd
from utils import config
from typing import Union

from shapely import wkb


# save results to .cache folder, so database isn't queried each time

# cache = joblib.Memory(".cache")  # , verbose=0


def get_connection() -> sqlalchemy.engine.Engine:
    """Creates a SQLAlchemy engine to access the database.

    Returns
    -------
    sqlalchemy.Engine
    """
    return sqlalchemy.create_engine(
        "postgresql://{username}:{password}@{url}:{port}/{db_name}".format(
            username=config.username,
            url=config.url,
            password=config.password,
            db_name=config.db_name,
            port=config.port,
        )
    )


def get_query(name: str) -> str:
    """Load the text of a SQL query saved in the sql/ folder.

    Parameters
    ----------
    name : str
        Name of the query without the sql/ part or the file extension.

    Returns
    -------
    str
        SQL query.
    """
    with open(f"sql/{name}.sql", "r") as f:
        return f.read()


def run_sql(filename: str, **kwargs) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Wrapper for run_sql_query.
    Loads string from specified file and call cached function,
    so cache gets updated on sql file edits.

    ----------
    filename : str
        Name of the file within the sql/ folder without file extension.

    Returns
    -------
    df
        DataFrame of results.
    """
    query = get_query(filename)
    df = run_sql_query(query, **kwargs)
    return df


#@cache.cache
def run_sql_query(
    query: str, geom_col: str = None, crs: int = None, **kwargs
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Execute an sql string and cache the result in joblib.Memory

    Parameters
    ----------
    query : str
        String representation of an SQL Query.
        May include template variables like %(schema)s.

    **kwargs : arguments passed to pd.read_sql.
        Typically index_col to set an index column out of the box,
        params for SQL templates like params = {'schema': 'osm'}

    Returns
    -------
    df : pd.DataFrame
    """
    conn = get_connection()
    print('SQL query: ')
    print(query)
    df = pd.read_sql(query, conn, **kwargs)

    if geom_col != None:
        # if parameters geom_col is specified, process the well-known-binary column 
        # and return a GeoDataFrame for convenience
        gdf = gpd.GeoDataFrame(df, geometry=wkb.loads(df[geom_col]), crs=crs)
        # drop wkb ghibberish
        gdf.drop(columns=[geom_col,], inplace=True)
        return gdf
    else:
        # just return the pandas DataFrame
        return df
