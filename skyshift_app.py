import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from astropy.io import fits
from skyshift_sequence import render_sequence


st.set_page_config(
    page_title="SkyShift | SPHEREx Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
DATA_FOLDER = BASE / "data" / "real_spherex"

LOGO_PATH = BASE / "skyshift_logo.png"

if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH), size="large")

from skyshift_search import (
    SkyPosition,
    build_search_query,
    search_spherex,
    validate_position,
)


@st.cache_data(show_spinner=False)
def inspect_fits(file_bytes):
    with fits.open(io.BytesIO(file_bytes), memmap=False) as hdul:
        candidates = [
            index
            for index, hdu in enumerate(hdul)
            if isinstance(
                hdu,
                (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU),
            )
            and hdu.data is not None
            and hdu.data.ndim == 2
        ]

        if not candidates:
            raise ValueError("No usable 2D image extension was found.")

        selected = next(
            (
                index
                for index in candidates
                if hdul[index].name.upper() == "IMAGE"
            ),
            candidates[0],
        )

        hdu = hdul[selected]
        image = np.array(hdu.data, dtype=float, copy=True)
        header = hdu.header.copy()
        primary = hdul[0].header.copy()

        def value(key, default="Not available"):
            return header.get(key, primary.get(key, default))

        metadata = {
            "Observation UTC": str(value("DATE-OBS")),
            "Detector": str(value("DETECTOR")),
            "Units": str(value("BUNIT")),
            "Selected extension": hdu.name,
            "Columns": int(image.shape[1]),
            "Rows": int(image.shape[0]),
            "Finite pixels (%)": round(
                float(np.isfinite(image).mean() * 100),
                2,
            ),
            "Wavelength lookup present": "WCS-WAVE" in hdul,
            "Flags present": "FLAGS" in hdul,
            "Variance present": "VARIANCE" in hdul,
        }

        return {
            "image": image,
            "metadata": metadata,
            "header": header.tostring(
                sep="\n",
                endcard=False,
                padding=False,
            ),
        }


with st.sidebar:
    st.title("SkyShift")
    st.caption("SPHEREx sky-image explorer")

    st.divider()

    st.subheader("Image source")

    source = st.radio(
        "Choose an input",
        ["Local project files", "Upload FITS files"],
    )

    st.divider()

    st.subheader("Display")

    color_map = st.selectbox(
        "Color map",
        ["gray", "inferno", "viridis"],
    )

    contrast = st.slider(
        "Display percentile limits",
        min_value=0.0,
        max_value=100.0,
        value=(1.0, 99.0),
        step=0.5,
    )

    st.caption(
        "Contrast changes affect the display only, "
        "not the stored scientific data."
    )

    st.divider()

    st.caption(
        "Prototype: coordinate search, cutout retrieval, "
        "FITS inspection, reprojection, and exploratory "
        "playback are implemented. Scientific validation "
        "is ongoing."
)


if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=360)
else:
    st.title("SkyShift")

st.subheader("Explore the changing infrared sky")


st.write(
    "Inspect SPHEREx observations, understand their metadata, "
    "and prepare to compare the same sky region over time."
)

st.info(
    "Explore SPHEREx observations by sky position and time. "
    "Displaying or animating images does not establish "
    "motion, variability, or an object's identity."
    
)

files = {}

if source == "Local project files":
    local_paths = (
        sorted(DATA_FOLDER.glob("*.fits"))
        if DATA_FOLDER.exists()
        else []
    )

    if local_paths:
        selected_paths = st.multiselect(
            "Choose downloaded FITS files",
            options=local_paths,
            default=local_paths[:2],
            format_func=lambda path: path.name,
        )

        for path in selected_paths:
            try:
                files[path.name] = path.read_bytes()
            except OSError as error:
                st.error(f"Could not read {path.name}: {error}")
    else:
        st.warning(
            "No FITS files were found in data/real_spherex. "
            "Choose Upload FITS files in the sidebar instead."
        )

else:
    uploaded = st.file_uploader(
        "Upload one or more FITS observations",
        type=["fits", "fit", "fts"],
        accept_multiple_files=True,
    )

    for index, item in enumerate(uploaded or []):
        label = f"{index + 1}. {item.name}"
        files[label] = item.getvalue()


records = {}

for name, content in files.items():
    try:
        records[name] = inspect_fits(content)
    except Exception as error:
        st.error(
            f"{name}: {type(error).__name__}: {error}"
        )


viewer_tab, search_tab, sequence_tab, metadata_tab, guide_tab = st.tabs(
    [
        "Explore images",
        "Coordinate search",
        "Aligned sequence",
        "Observation metadata",
        "Guide and project scope",
    ]
)  

with viewer_tab:
    if not records:
        st.write(
            "Select local files or upload FITS observations to begin."
        )

    else:
        chosen = st.selectbox(
            "Observation to inspect",
            options=list(records),
        )

        record = records[chosen]
        image = record["image"]
        metadata = record["metadata"]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Observation UTC",
            metadata["Observation UTC"],
        )

        col2.metric(
            "Detector",
            metadata["Detector"],
        )

        col3.metric(
            "Finite pixels",
            f"{metadata['Finite pixels (%)']:.2f}%",
        )

        finite = image[np.isfinite(image)]

        if finite.size == 0:
            st.error("This image contains no finite pixels.")

        else:
            low, high = np.percentile(finite, contrast)

            if high <= low:
                low = float(finite.min())
                high = float(finite.max())

            if high <= low:
                high = low + 1.0

            fig, ax = plt.subplots(figsize=(8, 6))

            plotted = ax.imshow(
                np.ma.masked_invalid(image),
                origin="lower",
                cmap=color_map,
                vmin=low,
                vmax=high,
                interpolation="nearest",
            )

            ax.set_xlabel("Pixel column")
            ax.set_ylabel("Pixel row")
            ax.set_title(
                f"Detector {metadata['Detector']} | "
                f"{metadata['Observation UTC']}"
            )

            fig.colorbar(
                plotted,
                ax=ax,
                label=metadata["Units"],
            )

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                f"{chosen} | "
                f"{image.shape[1]} columns × {image.shape[0]} rows"
            )

        st.warning(
            "Individual-image inspection only: images have not "
            "been aligned or approved for temporal comparison. "
            "Different observations may sample different wavelengths."
        )

        with st.expander("View science-image FITS header"):
            st.code(record["header"], language="text")
with search_tab:
    st.subheader("Find SPHEREx observations")

    st.caption(
        "Search image footprints around a custom sky position. "
        "Results are candidate observations, not confirmed "
        "detections of an astronomical object."
    )

    with st.form("coordinate_search_form"):
        col_ra, col_dec, col_radius = st.columns(3)

        ra_input = col_ra.text_input(
            "RA in decimal degrees",
            value="45.4488",
        )

        dec_input = col_dec.text_input(
            "Dec in decimal degrees",
            value="-7.2447",
        )

        radius_input = col_radius.number_input(
            "Search radius in degrees",
            min_value=0.01,
            max_value=1.0,
            value=0.10,
            step=0.01,
        )
        max_results = st.selectbox(
            "Maximum observation rows",
            options=[25, 100, 250, 500],
            index=0,
            help=(
                "Search metadata only. Increasing this does not "
                "download image files. Results are ordered "
                "from earliest to latest."
            ),
        )
        submitted = st.form_submit_button(
            "Search SPHEREx observations"
        )

    if submitted:
        st.session_state.pop("spherex_results", None)
        st.session_state.pop("spherex_search_center", None)
        st.session_state.pop("spherex_datalinks", None)

        try:
            position = validate_position(
                ra_input,
                dec_input,
            )

            st.code(
                build_search_query(
                    position,
                    radius_degrees=radius_input,
                    max_results=max_results,
                ),
                language="sql",
            )


            with st.spinner("Searching IRSA..."):
                csv_text = search_spherex(
                    position,
                    radius_degrees=radius_input,
                    max_results=max_results,
                )

            results = pd.read_csv(io.StringIO(csv_text))
            from astropy.time import Time

            results["Detector"] = (
                results["access_url"]
                .astype(str)
                .str.extract(r"/(D[1-6])(?:$|[&#])", expand=False)
                .fillna("Unknown")
            )

            mjd_values = pd.to_numeric(
                results["t_min"],
                errors="coerce",
            )

            results["Start time (UTC)"] = "Unavailable"

            valid_times = np.isfinite(mjd_values)

            if valid_times.any():
                results.loc[
                    valid_times,
                    "Start time (UTC)",
                ] = Time(
                    mjd_values[valid_times].to_numpy(),
                    format="mjd",
                    scale="utc",
                ).isot

            required = {"obs_id", "access_url"}

            if not required.issubset(results.columns):
                raise ValueError(
                    "The response is missing required columns: "
                    f"{sorted(required - set(results.columns))}"
                )

            st.session_state["spherex_results"] = results

            st.session_state["spherex_search_center"] = (
                position.ra_degrees,
                position.dec_degrees,
                radius_input,
            )

        except Exception as error:
            st.error(
                f"{type(error).__name__}: {error}"
            )

    results = st.session_state.get("spherex_results")

    if results is not None:
        ra, dec, radius = st.session_state[
            "spherex_search_center"
        ]

        st.caption(
            f"Custom field: RA {ra:.6f}°, "
            f"Dec {dec:.6f}°. Radius {radius:.2f}°."
        )

        if results.empty:
            st.warning(
                "The query completed with 0 observation rows. "
                "This alone does not establish a lack of coverage."
            )

        else:
            st.success(
                f"Returned {len(results)} observation rows."
            )

            st.caption(
                "Results are ordered from earliest to latest "
                "and may be truncated by the selected row limit. "
                "They are not a complete coverage inventory."
            )
            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download search results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="skyshift_spherex_search.csv",
                mime="text/csv",
                key="download_spherex_search_results",
            )

            st.subheader("Inspect a product's available links")

            selected_index = st.selectbox(
                "Choose an observation row",
                options=list(range(len(results))),
                format_func=lambda index: (
                    f"Row {index + 1} | "
                    f"{results.iloc[index]['Start time (UTC)']} | "
                    f"{results.iloc[index]['Detector']} | "
                    f"{results.iloc[index]['obs_id']}"
                ),
                key="spherex_selected_result",
            )

            selected = results.iloc[selected_index]
            access_url = str(selected["access_url"]).strip()

            st.code(access_url, language="text")

            if st.button(
                "Inspect DataLink",
                key="inspect_spherex_datalink",
            ):
                st.session_state.pop(
                    "spherex_datalinks",
                    None,
                )

                try:
                    import requests
                    from astropy.io.votable import parse_single_table
                    from urllib.parse import urlparse

                    parsed_url = urlparse(access_url)

                    if (
                        parsed_url.scheme != "https"
                        or parsed_url.hostname
                        != "irsa.ipac.caltech.edu"
                    ):
                        raise ValueError(
                            "Expected an HTTPS IRSA product link."
                        )

                    with st.spinner("Reading product links..."):
                        response = requests.get(
                            access_url,
                            timeout=45,
                        )

                        response.raise_for_status()

                        table = parse_single_table(
                            io.BytesIO(response.content)
                        ).to_table(use_names_over_ids=True)

                        links = table.to_pandas()

                        for column in links.columns:
                            links[column] = links[column].map(
                                lambda value: (
                                    value.decode(
                                        "utf-8",
                                        errors="replace",
                                    )
                                    if isinstance(value, bytes)
                                    else value
                                )
                            )

                    st.session_state["spherex_datalinks"] = {
                        "url": access_url,
                        "table": links,
                    }

                except Exception as error:
                    st.error(
                        f"{type(error).__name__}: {error}"
                    )

            link_record = st.session_state.get(
                "spherex_datalinks"
            )

            if (
                link_record is not None
                and link_record["url"] == access_url
            ):
                st.dataframe(
                    link_record["table"],
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download product-link metadata",
                    data=link_record["table"].to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="skyshift_product_links.csv",
                    mime="text/csv",
                    key="download_spherex_datalinks",
                )

                st.caption(
                    "These are service and product links. "
                    "No science-image FITS file has been downloaded."
                )
                st.subheader("Download a target-centered cutout")

                st.caption(
                    "Requests a small field around the search "
                    "coordinates, not the full detector image. "
                    "The request may fail if the requested region "
                    "does not contain usable image coverage."
                )

                if st.button(
                    "Download 0.10° cutout",
                    key="download_selected_cutout",
                ):
                    try:
                        import requests
                        from urllib.parse import urlparse

                        links = link_record["table"]

                        science_urls = []

                        for _, row in links.iterrows():
                            values = [
                                str(value).strip()
                                for value in row.values
                            ]

                            if "#this" not in values:
                                continue

                            for value in values:
                                parsed = urlparse(value)

                                if (
                                    parsed.scheme == "https"
                                    and parsed.hostname
                                    == "irsa.ipac.caltech.edu"
                                    and parsed.path.startswith(
                                        "/ibe/data/spherex/"
                                    )
                                    and parsed.path.lower().endswith(
                                        ".fits"
                                    )
                                ):
                                    science_urls.append(value)

                        science_urls = list(
                            dict.fromkeys(science_urls)
                        )

                        if len(science_urls) != 1:
                            raise ValueError(
                                "Expected exactly one on-premises "
                                "science FITS URL; found "
                                f"{len(science_urls)}."
                            )

                        download_folder = DATA_FOLDER
                        download_folder.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        original_name = Path(
                            urlparse(science_urls[0]).path
                        ).stem

                        destination = download_folder / (
                            f"{original_name}"
                            f"_ra{ra:.6f}_dec{dec:.6f}"
                            "_size0.10_cutout.fits"
                        )

                        temporary = destination.with_suffix(
                            ".fits.part"
                        )

                        size_limit = 20 * 1024 * 1024
                        total = 0

                        try:
                            with st.spinner(
                                "Requesting and validating cutout..."
                            ):
                                with requests.get(
                                    science_urls[0],
                                    params={
                                        "center": (
                                            f"{ra:.8f},{dec:.8f}"
                                        ),
                                        "size": "0.10",
                                    },
                                    stream=True,
                                    timeout=(15, 120),
                                ) as response:
                                    response.raise_for_status()

                                    with temporary.open("wb") as output:
                                        for chunk in response.iter_content(
                                            chunk_size=65536
                                        ):
                                            if not chunk:
                                                continue

                                            total += len(chunk)

                                            if total > size_limit:
                                                raise ValueError(
                                                    "Response exceeded "
                                                    "the 20 MB safety "
                                                    "limit. Download "
                                                    "was stopped."
                                                )

                                            output.write(chunk)

                                checked = inspect_fits(
                                    temporary.read_bytes()
                                )

                                if (
                                    checked["metadata"][
                                        "Finite pixels (%)"
                                    ] == 0
                                ):
                                    raise ValueError(
                                        "Cutout has no finite "
                                        "science-image pixels."
                                    )

                                temporary.replace(destination)

                        finally:
                            if temporary.exists():
                                temporary.unlink()

                        st.success(
                            "Saved a validated 2D FITS cutout: "
                            f"{destination.name} "
                            f"({total / 1024**2:.2f} MB)."
                        )

                        st.info(
                            "To open it, choose Local project files "
                            "in the sidebar, then select the new "
                            "cutout in Choose downloaded FITS files. "
                            "This validation does not yet establish "
                            "precise target coverage or wavelength "
                            "compatibility."
                        )

                    except Exception as error:
                        st.error(
                            f"{type(error).__name__}: {error}"
                        )
with sequence_tab:
    render_sequence(DATA_FOLDER)

with metadata_tab:
    if records:
        table = pd.DataFrame(
            [
                {"File": name, **record["metadata"]}
                for name, record in records.items()
            ]
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download inspection metadata",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="skyshift_inspection_metadata.csv",
            mime="text/csv",
            key="download_fits_inspection_metadata",
        )

        st.caption(
            "Metadata is read from the files. "
            "File origin is not independently verified by this viewer. "
            "Finite pixels alone do not establish scientific quality."
        )

    else:
        st.write("Load observations to see their metadata.")


with guide_tab:
    st.subheader("What this version does")

    st.markdown(
        """
        - Opens local or uploaded FITS observations.
        - Searches SPHEREx DR1 observation footprints by coordinates.
        - Resolves product links and retrieves small FITS cutouts.
        - Displays science images, headers, and observation metadata.
        - Provides contrast and color-map controls.
        - Reprojects selected cutouts onto a common sky grid.
        - Provides manual frame inspection and exploratory animation.
        - Exports metadata and an annotated animation.
        """
    )

    st.subheader("Current limitations and pending validation")

    st.markdown(
        """
        - Wavelength compatibility between selected observations is not yet verified.
        - Target identification and motion detection are not automated.
        - Quality flags and contamination are not yet fully assessed.
        - Validation on all five challenge targets is not complete.
        - Playback is exploratory, not a calibrated brightness comparison.
        """
    )


    st.subheader("Challenge validation targets")

    st.write(
        "Barnard's Star; Epsilon Indi B; WISE 0855; "
        "RW Aurigae; R Draconis."
    )

    st.caption(
        "These are planned validation targets. "
        "This page does not claim they have already been tested."
    )

    st.subheader("Scientific interpretation")

    st.write(
        "A visual difference can result from wavelength coverage, "
        "image alignment, noise, artifacts, or real sky changes. "
        "Candidate observations require further checking."
    )