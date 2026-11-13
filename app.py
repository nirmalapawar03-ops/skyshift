import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from skyshift_research import (
    compute_stcs,
    create_candidate_record,
    create_provenance_card,
    evaluate_image_quality,
    save_research_outputs,
)

def select_image_hdu(hdul):
    """
    Select the first HDU that contains a usable 2D image array.
    Prefers an HDU named IMAGE.
    """

    for hdu in hdul:
        data = hdu.data

        if data is None:
            continue

        if getattr(data, "ndim", 0) < 2:
            continue

        if hdu.name.upper() == "IMAGE":
            return hdu

    for hdu in hdul:
        data = hdu.data

        if data is None:
            continue

        if getattr(data, "ndim", 0) >= 2:
            return hdu

    return None


MODEL_FOLDER = Path(__file__).resolve().parent / "models"
SDSS_MODEL_PATH = MODEL_FOLDER / "sdss_classifier.joblib"
SDSS_METRICS_PATH = MODEL_FOLDER / "sdss_classifier_metrics.json"


@st.cache_resource
def load_sdss_classifier():
    if not SDSS_MODEL_PATH.exists():
        return None

    return joblib.load(SDSS_MODEL_PATH)



st.set_page_config(
    page_title="SkyShift",
    page_icon="🔭",
    layout="wide",
)

st.title("🔭 SkyShift")
st.caption("A sky-image explorer — beginner prototype")

st.warning(
    "Development mode: this page displays a simulated test image. "
    "It is not a real SPHEREx observation."
)

st.header("Our first image viewer")

project_folder = Path(__file__).resolve().parent
image_path = project_folder / "stage0_test.png"

if image_path.exists():
    st.image(
        str(image_path),
        caption="Simulated image created during Stage 0",
    )
else:
    st.error("The test image was not found beside app.py.")
    st.write("Expected location:", str(image_path))

with st.expander("What can this version do?"):
    st.write(
        "This first version displays a saved image in your browser. "
        "It does not yet search NASA data, track motion, "
        "or classify astronomical objects."
    )


st.divider()
st.header("Upload a test image")

st.caption(
    "PNG and JPG preview only. FITS support and scientific analysis "
    "will be added separately."
)

uploaded_file = st.file_uploader(
    "Choose an image from your computer",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    try:
        st.image(
            uploaded_file.getvalue(),
            caption=f"Uploaded image: {uploaded_file.name}",
        )
        st.info(
            "Preview only: this app has not verified the image's "
            "origin or identified any astronomical objects."
        )
    except Exception:
        st.error(
            "This file could not be displayed. "
            "Please try a valid PNG or JPG image."
        )

# FITS image viewer
import io

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder


st.divider()
st.header("FITS image viewer")

st.caption(
    "Upload a small FITS file containing a 2D image. "
    "This viewer does not classify objects or verify the file's origin."
)

fits_upload = st.file_uploader(
    "Choose a FITS image",
    type=["fits", "fit", "fts"],
    key="fits_upload",
)

if fits_upload is not None:
    if fits_upload.size > 20 * 1024 * 1024:
        st.error(
            "For this beginner prototype, please use a file under 20 MB."
        )
    else:
        try:
            with fits.open(
                io.BytesIO(fits_upload.getvalue()),
                memmap=False,
            ) as hdul:

                # List only extensions containing a 2D image.
                image_extensions = [
                    index
                    for index, hdu in enumerate(hdul)
                    if isinstance(
                        hdu,
                        (
                            fits.PrimaryHDU,
                            fits.ImageHDU,
                            fits.CompImageHDU,
                        ),
                    )
                    and hdu.data is not None
                    and hdu.data.ndim == 2
                ]

                if not image_extensions:
                    st.warning(
                        "No 2D image was found. "
                        "Tables and 3D image cubes are not supported yet."
                    )
                else:
                    selected_index = st.selectbox(
                        "Image extension to display",
                        options=image_extensions,
                        format_func=lambda index: (
                            f"{index}: {hdul[index].name}"
                        ),
                    )

                    image = np.array(
                        hdul[selected_index].data,
                        dtype=float,
                        copy=True,
                    )

                    header = hdul[selected_index].header.copy()
                    primary_header = hdul[0].header.copy()

                    is_simulated = (
                        primary_header.get("SIMULATE") is True
                        or header.get("SIMULATE") is True
                    )

                    if is_simulated:
                        st.warning(
                            "SIMULATED TEST DATA — "
                            "not a real astronomical observation."
                        )
                    else:
                        st.info(
                            "File origin has not been verified. "
                            "An uploaded FITS file is not automatically "
                            "SPHEREx data."
                        )

                    finite_pixels = image[np.isfinite(image)]

                    if finite_pixels.size == 0:
                        st.error("The image has no finite pixel values.")
                    else:
                        lower_percentile, upper_percentile = st.slider(
                            "Display contrast: percentile limits",
                               min_value=0.0,
                               max_value=100.0,
                               value=(1.0, 99.0),
                               step=0.5,
                        )

                        low, high = np.percentile(
                            finite_pixels,
                            [lower_percentile, upper_percentile],
                        )
                        if high <= low:
                            low = float(finite_pixels.min())
                            high = float(finite_pixels.max())

                        if high <= low:
                            high = low + 1

                        color_map = st.selectbox(
                            "Display colors",
                            ["gray", "inferno", "viridis"],
                        )

                        fig, ax = plt.subplots(figsize=(6, 5))
                        show_detections = st.checkbox(
                            "Detect point-like sources",
                            value=False,
                        )

                        detected_sources = None

                        if show_detections:
                            st.caption(
                                "Test settings: FWHM 4.7 pixels and "
                                "a threshold of 5 times the estimated noise. "
                                "These settings are tuned for our simulated image."
                            )

                            _, background, noise = sigma_clipped_stats(
                                image,
                                sigma=3.0,
                                mask=~np.isfinite(image),
                            )

                            if np.isfinite(noise) and noise > 0:
                                detection_image = np.where(
                                    np.isfinite(image),
                                    image - background,
                                    0.0,
                                )

                                finder = DAOStarFinder(
                                    fwhm=4.7,
                                    threshold=5.0 * noise,
                                    exclude_border=True,
                                )

                                detected_sources = finder(
                                    detection_image,
                                    mask=~np.isfinite(image),
                                )

                                if detected_sources is None:
                                    st.info("No point-like sources detected.")
                                else:
                                    st.write(
                                        "Detected point-like sources:",
                                        len(detected_sources),
                                    )

                                    positions = detected_sources[
                                        ["id", "x_centroid", "y_centroid"]
                                    ]

                                    st.dataframe(
                                        positions.to_pandas(),
                                        hide_index=True,
                                    )
                            else:
                                st.warning(
                                    "Could not estimate a positive noise level."
                                )

                        plotted_image = ax.imshow(
                            np.ma.masked_invalid(image),
                            origin="lower",
                            cmap=color_map,
                            vmin=low,
                            vmax=high,
                        )

                        ax.set_title(fits_upload.name)
                        if (
                            show_detections
                            and detected_sources is not None
                            and len(detected_sources) > 0
                        ):
                            ax.scatter(
                                detected_sources["x_centroid"],
                                detected_sources["y_centroid"],
                                s=250,
                                facecolors="none",
                                edgecolors="cyan",
                                linewidths=1.5,
                            )

                            for source in detected_sources:
                                ax.annotate(
                                    str(int(source["id"])),
                                    (
                                        float(source["x_centroid"]),
                                        float(source["y_centroid"]),
                                    ),
                                    xytext=(8, 8),
                                    textcoords="offset points",
                                    color="cyan",
                                )
                        ax.set_xlabel("Pixel column")
                        ax.set_ylabel("Pixel row")

                        fig.colorbar(
                            plotted_image,
                            ax=ax,
                            label=header.get(
                                "BUNIT",
                                "Pixel value — unit unspecified",
                            ),
                        )

                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                        st.write(
                            f"Image size: {image.shape[1]} columns "
                            f"× {image.shape[0]} rows"
                        )

                        st.caption(
                            "Colors represent pixel values, not visible-light "
                            "colors. The slider changes display limits only; "
                            "the stored image is unchanged."
                        )

                        with st.expander(
                            "View selected image header"
                        ):
                            st.code(
                                header.tostring(
                                    sep="\n",
                                    endcard=False,
                                    padding=False,
                                ),
                                language="text",
                            )

        except Exception as error:
            st.error("The FITS file could not be displayed.")
            st.code(str(error))

st.divider()
st.header("SDSS Catalog Classifier")
st.caption(
    "Demonstration classifier trained on the Kaggle SDSS17 "
    "stellar classification dataset."
)

sdss_model_bundle = load_sdss_classifier()

if sdss_model_bundle is None:
    st.warning(
        "No trained SDSS model found. Run "
        "`train_sdss_classifier.py` first."
    )
    st.stop()

sdss_pipeline = sdss_model_bundle["pipeline"]
sdss_features = sdss_model_bundle["features"]
sdss_labels = sdss_model_bundle["labels"]

st.info(sdss_model_bundle["scope"])

if SDSS_METRICS_PATH.exists():
    sdss_metrics = json.loads(
        SDSS_METRICS_PATH.read_text(encoding="utf-8")
    )

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)

    metric_col_1.metric(
        "Held-out accuracy",
        f"{sdss_metrics['accuracy']:.1%}",
    )

    metric_col_2.metric(
        "Balanced accuracy",
        f"{sdss_metrics['balanced_accuracy']:.1%}",
    )

    metric_col_3.metric(
        "Macro F1",
        f"{sdss_metrics['macro_f1']:.3f}",
    )

    st.caption(
        "Evaluated on 20,000 held-out SDSS catalog rows. "
        "These scores describe the model, not any new prediction."
    )
    report_path = (
        Path(__file__).resolve().parent
        / "models"
        / "sdss_classifier_report.txt"
    )

    matrix_path = (
        Path(__file__).resolve().parent
        / "models"
        / "sdss_classifier_confusion_matrix.csv"
    )

    if report_path.exists():
        with st.expander("Show held-out classification report"):
            st.code(
                report_path.read_text(encoding="utf-8"),
                language="text",
            )

            st.download_button(
                "Download classification report",
                data=report_path.read_text(encoding="utf-8"),
                file_name="sdss_classifier_report.txt",
                mime="text/plain",
            )

    if matrix_path.exists():
        with st.expander("Show confusion matrix"):
            confusion_df = pd.read_csv(
                matrix_path,
                index_col=0,
            )

            st.dataframe(
                confusion_df,
                use_container_width=True,
            )

            st.download_button(
                "Download confusion matrix",
                data=confusion_df.to_csv().encode("utf-8"),
                file_name="sdss_classifier_confusion_matrix.csv",
                mime="text/csv",
            )

st.subheader("Enter SDSS photometric measurements")

with st.form("sdss_prediction_form"):
    input_col_1, input_col_2, input_col_3, input_col_4, input_col_5 = (
        st.columns(5)
    )

    u_value = input_col_1.number_input(
        "u magnitude",
        min_value=0.0,
        max_value=40.0,
        value=18.0,
        step=0.01,
    )

    g_value = input_col_2.number_input(
        "g magnitude",
        min_value=0.0,
        max_value=40.0,
        value=17.5,
        step=0.01,
    )

    r_value = input_col_3.number_input(
        "r magnitude",
        min_value=0.0,
        max_value=40.0,
        value=17.0,
        step=0.01,
    )

    i_value = input_col_4.number_input(
        "i magnitude",
        min_value=0.0,
        max_value=40.0,
        value=16.8,
        step=0.01,
    )

    z_value = input_col_5.number_input(
        "z magnitude",
        min_value=0.0,
        max_value=40.0,
        value=16.6,
        step=0.01,
    )

    sdss_submitted = st.form_submit_button(
        "Classify object"
    )

if sdss_submitted:
    sdss_input = np.array(
        [[u_value, g_value, r_value, i_value, z_value]],
        dtype=float,
    )

    sdss_prediction = sdss_pipeline.predict(sdss_input)[0]
    sdss_probabilities = sdss_pipeline.predict_proba(sdss_input)[0]

    st.success(
        f"Predicted class: **{sdss_prediction}**"
    )

    sdss_result = pd.DataFrame(
        {
            "Class": sdss_labels,
            "Model score": sdss_probabilities,
        }
    ).sort_values(
        "Model score",
        ascending=False,
    )

    st.bar_chart(
        sdss_result.set_index("Class")
    )

    st.dataframe(
        sdss_result,
        use_container_width=True,
    )

    st.warning(
        "This is a model score, not a calibrated probability. "
        "This classifier identifies STAR, GALAXY, and QSO from "
        "SDSS photometry. It cannot detect asteroids and is not "
        "valid for SPHEREx image pixels."
    )

st.divider()
st.header("Project Status")

status_col_1, status_col_2, status_col_3 = st.columns(3)

status_col_1.success(
    "**Real data**\n\n"
    "SPHEREx FITS viewing and SDSS catalog classification."
)

status_col_2.warning(
    "**Simulated demonstration**\n\n"
    "Moving-source blink, difference image, and displacement."
)

status_col_3.info(
    "**Future work**\n\n"
    "Validated asteroid detection using an appropriate labeled "
    "NASA dataset."
)

with st.expander("What this app can and cannot claim"):
    st.markdown(
        """
        ### Demonstrated

        - Display and inspect a real SPHEREx FITS cutout.
        - Calculate basic image statistics and pixel-quality indicators.
        - Demonstrate moving-source concepts with simulated observations.
        - Train and evaluate a STAR/GALAXY/QSO classifier on the Kaggle SDSS17 catalog.
        - Show held-out metrics, a classification report, and a confusion matrix.

        ### Not claimed

        - This app does **not** currently detect asteroids from SPHEREx images.
        - The SDSS classifier does **not** classify SPHEREx pixels.
        - The SDSS classifier is not an asteroid detector.
        - Model scores are not calibrated probabilities.
        - The motion laboratory uses simulated observations, not a validated real moving-object pipeline.

        ### Next research step

        Use a properly labeled NASA dataset with repeated sky observations,
        known object associations, and compatible infrared measurements before
        training or evaluating an asteroid-detection model.
        """
    )

st.divider()
st.header("Candidate Screening — Experimental")
st.caption(
    "Compares two uploaded SPHEREx-style images of the same sky field. "
    "Outputs are screening candidates, not asteroid classifications."
)

st.warning(
    "This module is experimental. A high consistency score does not "
    "identify an asteroid, transient, or any specific object type."
)

with st.form("stcs_form"):
    st.subheader("Upload two aligned observations")

    reference_file = st.file_uploader(
        "Reference observation FITS",
        type=["fits", "fit"],
        key="stcs_reference",
    )

    comparison_file = st.file_uploader(
        "Comparison observation FITS",
        type=["fits", "fit"],
        key="stcs_comparison",
    )

    target_name = st.text_input(
        "Target or field label",
        value="Untitled field",
    )

    anomaly_sigma = st.slider(
        "Anomaly threshold in robust sigma",
        min_value=3.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
    )

    inject_test_source = st.checkbox(
        "Inject artificial test source into comparison image",
        value=False,
    )

    stcs_submitted = st.form_submit_button(
        "Run experimental screening"
    )

if stcs_submitted:
    if reference_file is None or comparison_file is None:
        st.error(
            "Upload both a reference and comparison FITS file."
        )
        st.stop()

    try:
        from astropy.io import fits

        with fits.open(reference_file, memmap=False) as reference_hdul:
            reference_hdu = select_image_hdu(reference_hdul)

            if reference_hdu is None:
                st.error(
                    "No usable 2D image array was found in the "
                    "reference FITS file."
                )
                st.stop()

            reference_image = np.asarray(
                reference_hdu.data,
                dtype=float,
            )
            reference_header = reference_hdu.header.copy()

            reference_hdu_info = {
                "selected_hdu_name": reference_hdu.name,
                "selected_hdu_index": reference_hdul.index_of(
                    reference_hdu.name
                )
                if reference_hdu.name in reference_hdul
                else 0,
                "hdu_count": len(reference_hdul),
                "image_shape": list(reference_image.shape),
                "finite_pixel_fraction": float(
                    np.isfinite(reference_image).mean()
                ),
            }

        with fits.open(comparison_file, memmap=False) as comparison_hdul:
            comparison_hdu = select_image_hdu(comparison_hdul)

            if comparison_hdu is None:
                st.error(
                    "No usable 2D image array was found in the "
                    "comparison FITS file."
                )
                st.stop()

            comparison_image = np.asarray(
                comparison_hdu.data,
                dtype=float,
            )
            comparison_header = comparison_hdu.header.copy()

            comparison_hdu_info = {
                "selected_hdu_name": comparison_hdu.name,
                "selected_hdu_index": comparison_hdul.index_of(
                    comparison_hdu.name
                )
                if comparison_hdu.name in comparison_hdul
                else 0,
                "hdu_count": len(comparison_hdul),
                "image_shape": list(comparison_image.shape),
                "finite_pixel_fraction": float(
                    np.isfinite(comparison_image).mean()
                ),
            }

        if reference_image.shape != comparison_image.shape:
            st.error(
                "The two images have different dimensions. "
                "Align or crop them before comparison."
            )
            st.stop()

        analysis_id = (
            "skyshift_"
            + time.strftime("%Y%m%d_%H%M%S")
        )

        reference_quality = evaluate_image_quality(
            reference_image
        )

        comparison_quality = evaluate_image_quality(
            comparison_image
        )
        if inject_test_source:
            test_y = reference_image.shape[0] // 2
            test_x = reference_image.shape[1] // 2

            finite_reference = reference_image[
                np.isfinite(reference_image)
            ]

            if finite_reference.size == 0:
                st.error(
                    "No valid reference pixels are available for "
                    "the injection test."
                )
                st.stop()

            background_level = float(
                np.nanmedian(finite_reference)
            )

            noise_level = float(
                1.4826
                * np.nanmedian(
                    np.abs(
                        finite_reference
                        - background_level
                    )
                )
            )

            injection_amplitude = max(
                10.0 * noise_level,
                1e-6,
            )

            comparison_image[test_y, test_x] = (
                comparison_image[test_y, test_x]
                + injection_amplitude
            )

            st.info(
                f"Injected artificial source at pixel "
                f"(x={test_x}, y={test_y}) with amplitude "
                f"{injection_amplitude:.6g}."
            )

        stcs_result = compute_stcs(
            reference_image,
            comparison_image,
            reference_quality,
            comparison_quality,
        )

        # Override the module threshold with the user-selected value.
        stcs_result["anomaly_threshold"] = anomaly_sigma

        reference_metadata = {
            "file_name": reference_file.name,
            "date_obs": reference_header.get("DATE-OBS", ""),
            "detector": reference_header.get("DETECTOR", ""),
            "wavelength": reference_header.get("WAVELEN", ""),
            "shape": list(reference_image.shape),
        }

        comparison_metadata = {
            "file_name": comparison_file.name,
            "date_obs": comparison_header.get("DATE-OBS", ""),
            "detector": comparison_header.get("DETECTOR", ""),
            "wavelength": comparison_header.get("WAVELEN", ""),
            "shape": list(comparison_image.shape),
        }

        candidate_record = create_candidate_record(
            analysis_id,
            target_name,
            reference_metadata,
            comparison_metadata,
            stcs_result,
        )

        provenance_card = create_provenance_card(
            analysis_id=analysis_id,
            analysis_type="spectral_temporal_screening",
            inputs={
                "reference_file": reference_file.name,
                "comparison_file": comparison_file.name,
                "target_name": target_name,
            },
            parameters={
                "anomaly_sigma": anomaly_sigma,
                "method": "median-centered robust difference",
                "quality_penalty": 0.25,
            },
            outputs={
                "stcs": stcs_result["stcs"],
                "verdict": stcs_result["verdict"],
                "anomalous_pixel_fraction": stcs_result[
                    "anomalous_pixel_fraction"
                ],
            },
            warnings=stcs_result["warnings"],
        )

        saved_paths = save_research_outputs(
            output_folder=Path("data") / "skyshift_research",
            analysis_id=analysis_id,
            provenance_card=provenance_card,
            candidate_record=candidate_record,
            difference_image=(
                comparison_image - reference_image
            ),
        )

        st.subheader("Uncertainty-first result")

        result_col_1, result_col_2, result_col_3 = st.columns(3)

        result_col_1.metric(
            "Screening verdict",
            stcs_result["verdict"],
        )
        with st.expander("Show FITS HDU selection"):
            st.write(
                {
                    "reference": reference_hdu_info,
                    "comparison": comparison_hdu_info,
                }
            )
        with st.expander("Show difference-image diagnostics"):
            st.write(
                {
                    "difference_median": stcs_result.get(
                        "difference_median"
                    ),
                    "difference_mad": stcs_result.get(
                        "difference_mad"
                    ),
                    "valid_pixel_count": stcs_result.get(
                        "valid_pixel_count"
                    ),
                    "anomalous_pixel_count": stcs_result.get(
                        "anomalous_pixel_count"
                    ),
                    "warnings": stcs_result.get(
                        "warnings"
                    ),
                }
            )

        result_col_2.metric(
            "STCS",
            (
                f"{stcs_result['stcs']:.3f}"
                if np.isfinite(stcs_result["stcs"])
                else "Not available"
            ),
        )

        result_col_3.metric(
            "Anomalous pixels",
            f"{stcs_result['anomalous_pixel_fraction']:.2%}",
        )

        quality_col_1, quality_col_2 = st.columns(2)

        quality_col_1.metric(
            "Reference quality",
            reference_quality["quality_flag"],
        )

        quality_col_2.metric(
            "Comparison quality",
            comparison_quality["quality_flag"],
        )

        st.caption(
            "STCS is a relative screening score, not a probability, "
            "confidence value, or object classification."
        )

        difference_image = comparison_image - reference_image

        valid_difference = difference_image[
            np.isfinite(difference_image)
        ]

        if valid_difference.size > 0:
            low, high = np.percentile(
                valid_difference,
                [1, 99],
            )

            st.image(
                np.clip(
                    (difference_image - low)
                    / max(high - low, 1e-9),
                    0,
                    1,
                ),
                caption="Difference image: comparison minus reference",
                use_container_width=True,
            )

        with st.expander("Show provenance and reproducibility card"):
            st.json(provenance_card)

            st.download_button(
                "Download provenance card",
                data=json.dumps(
                    provenance_card,
                    indent=2,
                    default=str,
                ),
                file_name=f"{analysis_id}_provenance.json",
                mime="application/json",
            )

        with st.expander("Show candidate record"):
            st.json(candidate_record)

            st.download_button(
                "Download candidate record",
                data=json.dumps(
                    candidate_record,
                    indent=2,
                    default=str,
                ),
                file_name=f"{analysis_id}_candidate.json",
                mime="application/json",
            )

        st.info(
            "Saved reproducible outputs:\n\n"
            f"- `{saved_paths['provenance_path']}`\n"
            f"- `{saved_paths['candidate_path']}`\n"
            f"- `{saved_paths['difference_path']}`"
        )

    except Exception as error:
        st.error(
            f"Candidate screening failed: "
            f"{type(error).__name__}: {error}"
        )