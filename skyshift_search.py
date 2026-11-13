import io
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import pandas as pd
import requests


IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP/sync"

DEFAULT_COLUMNS = """
    obs_collection,
    obs_id,
    t_min,
    t_max,
    s_ra,
    s_dec,
    instrument_name,
    dataproduct_type,
    access_url,
    target_name
"""


@dataclass
class SkyPosition:
    ra_degrees: float
    dec_degrees: float
    source: str = "User-entered coordinates"


def validate_position(ra_degrees, dec_degrees):
    try:
        ra = float(ra_degrees)
        dec = float(dec_degrees)
    except (TypeError, ValueError):
        raise ValueError("RA and Dec must be decimal degrees.")

    if not math.isfinite(ra) or not math.isfinite(dec):
        raise ValueError("RA and Dec must be finite numbers.")

    if not 0 <= ra < 360:
        raise ValueError(
            "RA must be at least 0 and less than 360 degrees."
        )

    if not -90 <= dec <= 90:
        raise ValueError(
            "Dec must be between -90 and 90 degrees."
        )

    return SkyPosition(ra, dec)


def build_search_query(
    position,
    radius_degrees=0.1,
    max_results=25,
):
    position = validate_position(
        position.ra_degrees,
        position.dec_degrees,
    )

    radius = float(radius_degrees)

    if not math.isfinite(radius) or not 0 < radius <= 1:
        raise ValueError(
            "Search radius must be greater than 0 "
            "and no more than 1 degree."
        )

    limit = int(max_results)

    if limit != float(max_results) or not 1 <= limit <= 1000:
        raise ValueError(
            "Maximum results must be an integer from 1 to 1000."
        )

    where = (
        "obs_collection = 'spherex_dr1' AND "
        "1=INTERSECTS("
        "s_region, "
        f"CIRCLE('ICRS', {position.ra_degrees:.8f}, "
        f"{position.dec_degrees:.8f}, {radius:.8f}))"
    )

    return (
        f"SELECT TOP {limit} "
        f"{DEFAULT_COLUMNS} "
        "FROM ivoa.ObsCore "
        f"WHERE {where} "
        "ORDER BY t_min ASC"
    )


def search_spherex(
    position,
    radius_degrees=0.1,
    timeout=45,
    max_results=25,
):
    query = build_search_query(
        position,
        radius_degrees=radius_degrees,
        max_results=max_results,
    )

    response = requests.get(
        IRSA_TAP_URL,
        params={
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": query,
        },
        timeout=timeout,
    )

    response.raise_for_status()

    text = response.text.lstrip("\ufeff \t\r\n")

    if text.startswith("<"):
        try:
            root = ET.fromstring(text)

            errors = [
                (element.text or "").strip()
                for element in root.iter()
                if element.tag.split("}")[-1] == "INFO"
                and element.attrib.get("name") == "QUERY_STATUS"
                and element.attrib.get("value") == "ERROR"
            ]

            message = "; ".join(errors) or text[:1000]

        except ET.ParseError:
            message = text[:1000]

        raise ValueError(
            f"IRSA returned XML instead of CSV: {message}"
        )

    try:
        table = pd.read_csv(io.StringIO(text))
    except Exception as error:
        raise ValueError(
            "IRSA's response could not be read as CSV."
        ) from error

    required = {"obs_id", "t_min", "access_url"}
    missing = required - set(table.columns)

    if missing:
        raise ValueError(
            f"Search response is missing columns: {sorted(missing)}"
        )

    return text