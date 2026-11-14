import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from astropy.io import fits
from skyshift_sequence import render_sequence


def evaluate_image_quality(image):
    finite = image[np.isfinite(image)]

    if finite.size == 0:
        return {
            "valid": False,
            "finite_pixels": 0,
            "finite_percentage": 0.0,
            "status": "NO_VALID_DATA",
        }

    finite_percentage = float(np.isfinite(image).mean() * 100)

    if finite_percentage < 50:
        status = "LOW_VALID_PIXEL_COVERAGE"
    else:
        status = "OK"

    return {
        "valid": True,
        "finite_pixels": int(finite.size),
        "finite_percentage": finite_percentage,
        "status": status,
    }


def compute_stcs(reference_image, comparison_image, reference_quality, comparison_quality):
    if reference_image.shape != comparison_image.shape:
        return {
            "stcs": 0.0,
            "anomaly_pixels": 0,
            "anomaly_percentage": 0.0,
            "verdict": "INCOMPATIBLE_DIMENSIONS",
            "valid_pixels": 0,
        }

    valid_mask = np.isfinite(reference_image) & np.isfinite(comparison_image)
    valid_pixels = int(valid_mask.sum())

    if valid_pixels == 0:
        return {
            "stcs": 0.0,
            "anomaly_pixels": 0,
            "anomaly_percentage": 0.0,
            "verdict": "INSUFFICIENT_DATA",
            "valid_pixels": 0,
        }

    difference = comparison_image[valid_mask] - reference_image[valid_mask]
    median_difference = float(np.median(difference))
    centered_difference = difference - median_difference

    mad = float(np.median(np.abs(centered_difference)))
    robust_sigma = 1.4826 * mad

    if robust_scale := max(robust_sigma, 1e-12):
        normalized_difference = centered_difference / robust_scale
    else:
        normalized_difference = centered_difference

    anomaly_threshold = 5.0
    anomaly_mask = np.abs(normalized_difference) > anomaly_threshold
    anomaly_pixels = int(anomaly_mask.sum())
    anomaly_percentage = float(anomaly_pixels / valid_pixels * 100)

    stcs = float(np.mean(np.abs(normalized_difference)))

    if reference_quality["status"] != "OK" or comparison_quality["status"] != "OK":
        verdict = "REVIEW_WITH_LOW_CONFIDENCE"
    elif anomaly_pixels == 0:
        verdict = "NO_DIFFERENCE"
    else:
        verdict = "REVIEW"

    return {
        "stcs": stcs,
        "anomaly_pixels": anomaly_pixels,
        "anomaly_percentage": anomaly_percentage,
        "verdict": verdict,
        "valid_pixels": valid_pixels,
        "median_difference": median_difference,
        "robust_sigma": robust_sigma,
        "anomaly_threshold": anomaly_threshold,
    }


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
        flags = None
        variance = None
        wavelength = None
        bandwidth = None

        for extra_hdu in hdul:
            if extra_hdu.name.upper() == "FLAGS":
                if extra_hdu.data is not None and extra_hdu.data.ndim == 2:
                    flags = np.array(extra_hdu.data, copy=True)

            if extra_hdu.name.upper() == "VARIANCE":
                if extra_hdu.data is not None and extra_hdu.data.ndim == 2:
                    variance = np.array(
                        extra_hdu.data,
                        dtype=float,
                        copy=True,
                    )

            if extra_hdu.name.upper() == "WCS-WAVE":
                try:
                    wave_table = extra_hdu.data

                    wave_x = np.asarray(wave_table["X"], dtype=float)
                    wave_y = np.asarray(wave_table["Y"], dtype=float)
                    wave_values = np.asarray(
                        wave_table["VALUES"],
                        dtype=float,
                    )

                    if (
                        wave_x.ndim == 1
                        and wave_y.ndim == 1
                        and wave_values.ndim == 2
                        and wave_values.shape[1] == 2
                    ):
                        wavelength = wave_values[:, 0]
                        bandwidth = wave_values[:, 1]

                except Exception:
                    wavelength = None
                    bandwidth = None        
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
            "flags": flags,
            "variance": variance,
            "wavelength": wavelength,
            "bandwidth": bandwidth,
            "metadata": metadata,
            "header": header.tostring(
                sep="\n",
                endcard=False,
                padding=False,
            ),
        }


def estimate_pixel_wavelength(record, row, column):
    wavelength = record.get("wavelength")
    bandwidth = record.get("bandwidth")

    if wavelength is None or bandwidth is None:
        return None, None

    with fits.open(
        io.BytesIO(st.session_state.get(f"fits_bytes_{record['file_name']}")),
        memmap=False,
    ) as hdul:
        wave_hdu = next(
            (
                hdu
                for hdu in hdul
                if hdu.name.upper() == "WCS-WAVE"
            ),
            None,
        )

        if wave_hdu is None or wave_hdu.data is None:
            return None, None

        wave_table = wave_hdu.data

        wave_x = np.asarray(wave_table["X"], dtype=float)
        wave_y = np.asarray(wave_table["Y"], dtype=float)
        wave_values = np.asarray(wave_table["VALUES"], dtype=float)

        central_wavelengths = wave_values[:, 0]
        bandwidths = wave_values[:, 1]

        x_distance = np.abs(wave_x - column)
        y_distance = np.abs(wave_y - row)

        x_order = np.argsort(x_distance)[:2]
        y_order = np.argsort(y_distance)[:2]

        nearby_x = wave_x[x_order]
        nearby_y = wave_y[y_order]

        nearby_values = central_wavelengths[
            np.ix_(y_order, x_order)
        ]

        x_weights = 1.0 / np.maximum(x_distance[x_order], 1e-12)
        y_weights = 1.0 / np.maximum(y_distance[y_order], 1e-12)

        weight_grid = np.outer(y_weights, x_weights)
        weight_sum = weight_grid.sum()

        estimated_wavelength = float(
            (nearby_values * weight_grid).sum() / weight_sum
        )

        nearby_bandwidths = bandwidths[
            np.ix_(y_order, x_order)
        ]

        estimated_bandwidth = float(
            (nearby_bandwidths * weight_grid).sum() / weight_sum
        )

        return estimated_wavelength, estimated_bandwidth

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


header_left, header_right = st.columns([1, 3], vertical_alignment="center")

with header_left:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=170)

with header_right:
    st.markdown(
        """
        <h1 style='margin-bottom:0px;'>SkyShift</h1>
        <p style='margin-top:4px; color:#8ea9c1; font-size:1.05rem;'>
        Explore the changing infrared sky
        </p>
        """,
        unsafe_allow_html=True,
    )


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
        record = inspect_fits(content)
        record["file_name"] = name
        st.session_state[f"fits_bytes_{name}"] = content
        records[name] = record

    except Exception as error:
        st.error(
            f"{name}: {type(error).__name__}: {error}"
        )


viewer_tab, search_tab, sequence_tab, metadata_tab, screening_tab, guide_tab = st.tabs(
    [
        "Explore images",
        "Coordinate search",
        "Aligned sequence",
        "Observation metadata",
        "Experimental screening",
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



with screening_tab:
    st.subheader("Experimental screening")
    st.caption(
        "Compare two observations of the same sky region using a robust, "
        "median-centred difference analysis. This is an experimental "
        "screening workflow, not an automated discovery system."
    )

    if len(records) < 2:
        st.warning(
            "Select or upload at least two FITS observations to run screening."
        )

    else:
        col_reference, col_comparison = st.columns(2)

        with col_reference:
            reference_name = st.selectbox(
                "Reference observation",
                options=list(records),
                key="screening_reference",
            )

        with col_comparison:
            comparison_name = st.selectbox(
                "Comparison observation",
                options=list(records),
                index=1,
                key="screening_comparison",
            )

        reference_record = records[reference_name]
        comparison_record = records[comparison_name]

        reference_image = reference_record["image"]
        comparison_image = comparison_record["image"]

        st.session_state["reference_image"] = reference_image

        st.info(
            f"Reference: {reference_name} | "
            f"{reference_image.shape[1]} × {reference_image.shape[0]} pixels\n\n"
            f"Comparison: {comparison_name} | "
            f"{comparison_image.shape[1]} × {comparison_image.shape[0]} pixels"
        )

        if reference_image.shape != comparison_image.shape:
            st.error(
                "The selected observations have different image dimensions. "
                "A direct pixel-based comparison requires matching dimensions."
            )

        else:
            if st.button(
                "Run experimental screening",
                use_container_width=True,
            ):
                reference_quality = evaluate_image_quality(reference_image)
                comparison_quality = evaluate_image_quality(comparison_image)

                st.session_state["screening_result"] = compute_stcs(
                    reference_image,
                    comparison_image,
                    reference_quality,
                    comparison_quality,
                )

                st.session_state["screening_quality"] = {
                    "reference": reference_quality,
                    "comparison": comparison_quality,
                }

            screening_result = st.session_state.get("screening_result")

            if screening_result is not None:
                quality = st.session_state.get("screening_quality", {})

                metric_one, metric_two, metric_three = st.columns(3)

                with metric_one:
                    st.metric(
                        "STCS",
                        f"{screening_result['stcs']:.6f}",
                    )

                with metric_two:
                    st.metric(
                        "Anomaly pixels",
                        f"{screening_result['anomaly_pixels']}",
                    )

                with metric_three:
                    st.metric(
                        "Anomaly percentage",
                        f"{screening_result['anomaly_percentage']:.4f}%",
                    )

                st.info(
                    f"Review flag: **{screening_result['verdict']}**"
                )

                st.markdown("#### Technical details")

                st.write(
                    f"Valid pixels compared: "
                    f"{screening_result['valid_pixels']}"
                )

                st.write(
                    f"Median difference: "
                    f"{screening_result.get('median_difference', 0):.6g}"
                )

                st.write(
                    f"Robust sigma (MAD-based): "
                    f"{screening_result.get('robust_sigma', 0):.6g}"
                )

                st.write(
                    f"Anomaly threshold: "
                    f"{screening_result.get('anomaly_threshold', 0):.1f} "
                    "robust sigma"
                )

                st.write(
                    f"Reference quality: "
                    f"{quality.get('reference', {}).get('status', 'Not assessed')}"
                )

                st.write(
                    f"Comparison quality: "
                    f"{quality.get('comparison', {}).get('status', 'Not assessed')}"
                )

                st.caption(
                    "A review flag indicates that closer expert review may be "
                    "needed. It does not establish a moving object, variability, "
                    "or an astronomical discovery."
                )

                st.markdown("### Candidate Review Report")

                report_col1, report_col2 = st.columns(2)

                with report_col1:
                    st.markdown("#### Screening summary")

                    st.write(
                        f"**Review status:** {screening_result['verdict']}"
                    )

                    st.write(
                        f"**STCS:** {screening_result['stcs']:.6f}"
                    )

                    st.write(
                        f"**Anomalous pixels:** "
                        f"{screening_result['anomaly_pixels']} / "
                        f"{screening_result['valid_pixels']}"
                    )

                    st.write(
                        f"**Anomaly percentage:** "
                        f"{screening_result['anomaly_percentage']:.4f}%"
                    )

                    st.write(
                        f"**Reference quality:** "
                        f"{quality.get('reference', {}).get('status', 'Not assessed')}"
                    )

                    st.write(
                        f"**Comparison quality:** "
                        f"{quality.get('comparison', {}).get('status', 'Not assessed')}"
                    )

                with report_col2:
                    st.markdown("#### Observation compatibility")

                    reference_parts = reference_name.split("_")
                    comparison_parts = comparison_name.split("_")

                    reference_band = (
                        reference_parts[2]
                        if len(reference_parts) > 2
                        else "Unknown"
                    )

                    comparison_band = (
                        comparison_parts[2]
                        if len(comparison_parts) > 2
                        else "Unknown"
                    )

                    reference_detector = (
                        reference_parts[4]
                        if len(reference_parts) > 4
                        else "Unknown"
                    )

                    comparison_detector = (
                        comparison_parts[4]
                        if len(comparison_parts) > 4
                        else "Unknown"
                    )

                    st.write(
                        f"**Same band:** "
                        f"{'Yes' if reference_band == comparison_band else 'No'} "
                        f"({reference_band} vs {comparison_band})"
                    )

                    st.write(
                        f"**Same detector:** "
                        f"{'Yes' if reference_detector == comparison_detector else 'No'} "
                        f"({reference_detector} vs {comparison_detector})"
                    )

                    st.write(
                        "**Same image dimensions:** "
                        f"{'Yes' if reference_image.shape == comparison_image.shape else 'No'}"
                    )

                    st.write("**Temporal validation:** Pending")

                    st.write("**WCS alignment validation:** Pending")

                    st.write("**Calibration validation:** Pending")

                st.markdown("#### Recommended next checks")

                st.markdown(
                    """
                    - Verify observation times and wavelength coverage.
                    - Inspect the difference image and anomalous pixel locations.
                    - Check background variation, detector artifacts, and saturation.
                    - Perform astrometric alignment before scientific interpretation.
                    - Compare results with independent observations or catalogs.
                    """
                )

                st.caption(
                    "This report is a structured screening record. It does not "
                    "establish a moving object, variability, or an astronomical "
                    "discovery."
                )

                st.markdown("#### Difference and anomaly map")

                valid_mask = (
                    np.isfinite(reference_image)
                    & np.isfinite(comparison_image)
                )

                difference_image = np.full(
                    reference_image.shape,
                    np.nan,
                    dtype=float,
                )

                difference_image[valid_mask] = (
                    comparison_image[valid_mask]
                    - reference_image[valid_mask]
                )

                finite_differences = difference_image[valid_mask]

                median_difference = float(
                    np.median(finite_differences)
                )

                mad = float(
                    np.median(
                        np.abs(
                            finite_differences
                            - median_difference
                        )
                    )
                )

                robust_sigma = max(
                    1.4826 * mad,
                    1e-12,
                )

                normalized_difference = (
                    difference_image - median_difference
                ) / robust_sigma

                anomaly_mask = (
                    np.abs(normalized_difference)
                    > screening_result["anomaly_threshold"]
                )

                fig_difference, ax_difference = plt.subplots(
                    figsize=(7, 5)
                )

                difference_plot = ax_difference.imshow(
                    normalized_difference,
                    origin="lower",
                    cmap="coolwarm",
                    vmin=-5,
                    vmax=5,
                    interpolation="nearest",
                )

                anomaly_rows, anomaly_columns = np.where(
                    anomaly_mask
                )

                if anomaly_rows.size > 0:
                    ax_difference.scatter(
                        anomaly_columns,
                        anomaly_rows,
                        facecolors="none",
                        edgecolors="lime",
                        s=110,
                        linewidths=1.8,
                        label="Anomalous pixels",
                    )

                    ax_difference.legend(
                        loc="upper right",
                        fontsize=8,
                    )

                ax_difference.set_xlabel("Pixel column")
                ax_difference.set_ylabel("Pixel row")
                ax_difference.set_title(
                    "Robust normalized difference"
                )

                fig_difference.colorbar(
                    difference_plot,
                    ax=ax_difference,
                    label="Robust sigma units",
                )

                fig_difference.tight_layout()
                st.pyplot(fig_difference)
                plt.close(fig_difference)

                if anomaly_rows.size > 0:
                    st.markdown("### Anomalous Pixel Inspector")
                    st.caption(
                        "Inspect each flagged pixel using its reference value, "
                        "comparison value, robust sigma, SPHEREx quality flags, "
                        "variance, and approximate wavelength."
                    )

                    inspector_results = []

                    for pixel_row, pixel_column in zip(
                        anomaly_rows,
                        anomaly_columns,
                    ):
                        reference_value = reference_image[
                            pixel_row,
                            pixel_column,
                        ]

                        comparison_value = comparison_image[
                            pixel_row,
                            pixel_column,
                        ]

                        pixel_difference = (
                            comparison_value - reference_value
                        )

                        pixel_sigma = (
                            pixel_difference - median_difference
                        ) / robust_sigma

                        reference_flags = None
                        comparison_flags = None

                        if reference_record.get("flags") is not None:
                            reference_flags = int(
                                reference_record["flags"][
                                    pixel_row,
                                    pixel_column,
                                ]
                            )

                        if comparison_record.get("flags") is not None:
                            comparison_flags = int(
                                comparison_record["flags"][
                                    pixel_row,
                                    pixel_column,
                                ]
                            )

                        reference_variance = None
                        comparison_variance = None

                        if reference_record.get("variance") is not None:
                            reference_variance = float(
                                reference_record["variance"][
                                    pixel_row,
                                    pixel_column,
                                ]
                            )

                        if comparison_record.get("variance") is not None:
                            comparison_variance = float(
                                comparison_record["variance"][
                                    pixel_row,
                                    pixel_column,
                                ]
                            )

                        if (
                            reference_flags is not None
                            and reference_flags != 0
                        ) or (
                            comparison_flags is not None
                            and comparison_flags != 0
                        ):
                            likely_category = (
                                "Artifact-suspect: SPHEREx quality flags are present"
                            )

                        elif (
                            reference_variance is not None
                            and comparison_variance is not None
                            and abs(pixel_difference)
                            > 3 * np.sqrt(
                                max(
                                    reference_variance
                                    + comparison_variance,
                                    1e-12,
                                )
                            )
                        ):
                            likely_category = (
                                "Potentially significant change: "
                                "needs catalog cross-match"
                            )

                        else:
                            likely_category = (
                                "Needs catalog cross-match and further validation"
                            )

                        estimated_wavelength, estimated_bandwidth = (
                            estimate_pixel_wavelength(
                                reference_record,
                                pixel_row,
                                pixel_column,
                            )
                        )


                        reference_variance = (
                            reference_variance
                            if reference_variance is not None
                            else np.nan
                        )

                        comparison_variance = (
                            comparison_variance
                            if comparison_variance is not None
                            else np.nan
                        )

                        combined_variance = (
                            reference_variance + comparison_variance
                            if np.isfinite(reference_variance)
                            and np.isfinite(comparison_variance)
                            else np.nan
                        )

                        combined_uncertainty = (
                            np.sqrt(max(combined_variance, 1e-12))
                            if np.isfinite(combined_variance)
                            else np.nan
                        )

                        signal_to_noise = (
                            abs(pixel_difference) / combined_uncertainty
                            if np.isfinite(combined_uncertainty)
                            and combined_uncertainty > 0
                            else np.nan
                        )

                        if (
                            reference_flags is not None
                            and reference_flags != 0
                        ) or (
                            comparison_flags is not None
                            and comparison_flags != 0
                        ):
                            source_likelihood = "Artifact-suspect"

                        elif (
                            np.isfinite(signal_to_noise)
                            and signal_to_noise >= 3
                        ):
                            source_likelihood = (
                                "Potential-change candidate"
                            )

                        else:
                            source_likelihood = (
                                "Low-confidence anomaly"
                            )

                        inspector_results.append(
                            {
                                 "Row": int(pixel_row),
                                "Column": int(pixel_column),
                                "Reference value": reference_value,
                                "Comparison value": comparison_value,
                                "Difference": pixel_difference,
                                "Robust sigma": pixel_sigma,
                                "Reference FLAGS": reference_flags,
                                "Comparison FLAGS": comparison_flags,
                                "Reference variance": reference_variance,
                                "Comparison variance": comparison_variance,
                                "Estimated wavelength (µm)": estimated_wavelength,
                                "Estimated bandwidth (µm)": estimated_bandwidth,
                                "Combined uncertainty": combined_uncertainty,
                                "Signal-to-noise estimate": signal_to_noise,
                                "Source likelihood": source_likelihood,
                                "Likely review category": likely_category,

                            }
                        )

                    inspector_table = pd.DataFrame(inspector_results)

                    st.dataframe(
                        inspector_table,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download anomalous pixel inspection table",
                        data=inspector_table.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="skyshift_anomalous_pixel_inspection.csv",
                        mime="text/csv",
                        key="download_anomalous_pixel_inspection",
                    )

                    candidate_pixels = inspector_table[
                        inspector_table["Source likelihood"]
                        == "Potential-change candidate"
                    ]

                    st.markdown("### Highest-Priority Change Candidates")
                    st.caption(
                        "These pixels have clean SPHEREx quality flags and a "
                        "difference exceeding the estimated combined measurement "
                        "uncertainty. They require catalog cross-matching and "
                        "additional observations before any scientific claim."
                    )

                    if candidate_pixels.empty:
                        st.info(
                            "No anomalous pixel currently meets the "
                            "potential-change-candidate criteria."
                        )

                    else:
                        st.dataframe(
                            candidate_pixels,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.warning(
                            "These are screening candidates, not confirmed "
                            "astronomical detections. A real source change "
                            "requires temporal validation, astrometric alignment, "
                            "calibration checks, and independent confirmation."
                        )

                    st.markdown("### SkyShift Decision Summary")

                    artifact_count = 0
                    potential_count = 0
                    crossmatch_count = 0

                    for item in inspector_results:
                        category = item["Likely review category"]

                        if category.startswith("Artifact-suspect"):
                            artifact_count += 1

                        elif category.startswith("Potentially significant"):
                            potential_count += 1

                        else:
                            crossmatch_count += 1

                    if artifact_count > 0:
                        decision = "ARTIFACT_SUSPECT"
                        decision_color = "orange"
                        decision_message = (
                            "One or more anomalous pixels have SPHEREx quality "
                            "flags. Treat these as possible instrumental or "
                            "processing effects until further validated."
                        )

                    elif potential_count > 0:
                        decision = "POTENTIAL_CHANGE_CANDIDATE"
                        decision_color = "green"
                        decision_message = (
                            "At least one anomalous pixel has clean quality flags "
                            "and a difference exceeding the estimated combined "
                            "measurement uncertainty. This requires catalog "
                            "cross-matching and additional validation."
                        )

                    elif screening_result["verdict"] == "NO_DIFFERENCE":
                        decision = "NO_FOLLOW_UP_NEEDED"
                        decision_color = "blue"
                        decision_message = (
                            "No anomalous pixels were detected above the "
                            "selected robust-sigma threshold."
                        )

                    else:
                        decision = "INSUFFICIENT_EVIDENCE"
                        decision_color = "gray"
                        decision_message = (
                            "Anomalies were detected, but the available quality "
                            "and uncertainty information is not sufficient for "
                            "a meaningful prioritization."
                        )

                    st.markdown(
                        f"""
                        <div style="
                            padding: 1rem;
                            border-radius: 0.75rem;
                            border: 1px solid {decision_color};
                            background-color: rgba(255,255,255,0.04);
                        ">
                            <h3 style="margin:0; color:{decision_color};">
                                {decision}
                            </h3>
                            <p style="margin-top:0.5rem;">
                                {decision_message}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    summary_col1, summary_col2, summary_col3 = st.columns(3)

                    with summary_col1:
                        st.metric(
                            "Artifact-suspect pixels",
                            artifact_count,
                        )

                    with summary_col2:
                        st.metric(
                            "Potential-change pixels",
                            potential_count,
                        )

                    with summary_col3:
                        st.metric(
                            "Cross-match required",
                            crossmatch_count,
                        )

                    st.markdown("#### Required before a scientific claim")

                    st.markdown(
                        """
                        - Confirm observation times and temporal separation.
                        - Validate WCS alignment and sky-coordinate matching.
                        - Check calibration, background, and detector effects.
                        - Cross-match pixel coordinates with astronomical catalogs.
                        - Compare additional independent observations.
                        - Obtain expert astronomical review.
                        """
                    )

                    st.caption(
                        "SkyShift provides a prioritized screening decision. "
                        "It does not confirm a moving object, variability, "
                        "or an astronomical discovery."
                    )

                    st.info(
                        "A clean SPHEREx flag and low variance do not confirm "
                        "a real astrophysical change. Cross-match the pixel "
                        "coordinates with astronomical catalogs and inspect "
                        "additional observations before classification."
                    )

                st.caption(
                    "Green circles mark pixels whose absolute normalized "
                    "difference exceeds the selected robust-sigma threshold. "
                    "This visualization supports review; it does not identify "
                    "an astronomical object."
                )

                review_report = {
                    "reference_file": reference_name,
                    "comparison_file": comparison_name,
                    "reference_band": reference_band,
                    "comparison_band": comparison_band,
                    "reference_detector": reference_detector,
                    "comparison_detector": comparison_detector,
                    "image_shape": list(reference_image.shape),
                    "screening_result": screening_result,
                    "quality": quality,
                    "same_band": reference_band == comparison_band,
                    "same_detector": (
                        reference_detector == comparison_detector
                    ),
                    "same_dimensions": (
                        reference_image.shape
                        == comparison_image.shape
                    ),
                    "temporal_validation": "Pending",
                    "wcs_alignment_validation": "Pending",
                    "calibration_validation": "Pending",
                    "recommended_checks": [
                        "Verify observation times and wavelength coverage",
                        "Inspect difference image and anomalous pixel locations",
                        "Check background variation, detector artifacts, and saturation",
                        "Perform astrometric alignment before scientific interpretation",
                        "Compare results with independent observations or catalogs",
                    ],
                    "interpretation": (
                        "Structured screening record. Does not establish a "
                        "moving object, variability, or an astronomical discovery."
                    ),
                }

                st.download_button(
                    "Download candidate review report",
                    data=json.dumps(
                        review_report,
                        indent=2,
                    ).encode("utf-8"),
                    file_name="skyshift_candidate_review_report.json",
                    mime="application/json",
                    key="download_candidate_review_report",
                )

        st.divider()
        st.markdown("### Validation & Comparison Dashboard")
        st.caption(
            "Run controlled tests to check whether the STCS workflow correctly "
            "distinguishes no-change and known-change cases."
        )

        validation_left, validation_right = st.columns(2)

        with validation_left:
            st.markdown("#### Null test")
            st.caption(
                "Uses the same reference image for both inputs."
            )

            run_null_test = st.button(
                "Run null test",
                use_container_width=True,
            )

        with validation_right:
            st.markdown("#### Injection test")
            st.caption(
                "Adds a controlled artificial change to a copy of the "
                "reference image."
            )

            run_injection_test = st.button(
                "Run injection test",
                use_container_width=True,
            )

        if run_null_test or run_injection_test:
            if "reference_image" not in st.session_state:
                st.warning(
                    "Please select a reference observation first."
                )
                st.stop()

            validation_reference = st.session_state["reference_image"]

            if run_null_test:
                validation_comparison = validation_reference.copy()
                validation_label = (
                    "Null test: identical reference and comparison"
                )

            if run_injection_test:
                validation_comparison = validation_reference.copy()

                finite_reference = validation_reference[
                    np.isfinite(validation_reference)
                ]

                if finite_reference.size == 0:
                    st.error(
                        "The reference image has no valid pixels, so the "
                        "injection test cannot run."
                    )
                    st.stop()

                robust_scale = np.median(
                    np.abs(
                        finite_reference
                        - np.median(finite_reference)
                    )
                )

                injection_value = max(
                    10 * 1.4826 * robust_scale,
                    1e-6,
                )

                center_row = validation_reference.shape[0] // 2
                center_column = validation_reference.shape[1] // 2

                validation_comparison[
                    center_row,
                    center_column,
                ] += injection_value

                validation_label = (
                    f"Injection test: controlled change of "
                    f"{injection_value:.6g} added to the central pixel"
                )

            validation_reference_quality = evaluate_image_quality(
                validation_reference
            )

            validation_comparison_quality = evaluate_image_quality(
                validation_comparison
            )

            validation_result = compute_stcs(
                validation_reference,
                validation_comparison,
                validation_reference_quality,
                validation_comparison_quality,
            )

            st.success(validation_label)

            validation_metric_one, validation_metric_two, validation_metric_three = st.columns(3)

            with validation_metric_one:
                st.metric(
                    "STCS",
                    f"{validation_result['stcs']:.6f}",
                )

            with validation_metric_two:
                st.metric(
                    "Anomaly pixels",
                    f"{validation_result['anomaly_pixels']}",
                )

            with validation_metric_three:
                st.metric(
                    "Anomaly percentage",
                    f"{validation_result['anomaly_percentage']:.4f}%",
                )

            st.info(
                f"Review flag: **{validation_result['verdict']}**"
            )

            st.markdown("#### Validation interpretation")

            if run_null_test:
                if validation_result["verdict"] == "NO_DIFFERENCE":
                    st.success(
                        "Expected result confirmed: identical inputs produced "
                        "no difference. This validates the workflow’s "
                        "no-change behaviour."
                    )

                else:
                    st.warning(
                        "Unexpected result: identical inputs did not produce "
                        "NO_DIFFERENCE. Review the selected image extension "
                        "and screening settings."
                    )

            if run_injection_test:
                if validation_result["stcs"] > 0:
                    st.success(
                        "The workflow detected the controlled change. This "
                        "demonstrates that STCS responds to a known "
                        "artificial difference."
                    )

                else:
                    st.warning(
                        "The controlled change was not reflected in the STCS "
                        "result. Check the selected image extension, valid "
                        "pixel coverage, and injection value."
                    )



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