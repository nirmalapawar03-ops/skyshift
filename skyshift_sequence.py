import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from reproject import reproject_interp


@st.cache_data(show_spinner=False)
def build_sequence(items):
    frames = []

    for name, content in items:
        with fits.open(io.BytesIO(content), memmap=False) as hdul:
            candidates = [
                hdu
                for hdu in hdul
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

            if not candidates:
                raise ValueError(f"{name}: no 2D image.")

            hdu = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.name.upper() == "IMAGE"
                ),
                candidates[0],
            )

            header = hdu.header
            primary = hdul[0].header

            date = header.get(
                "DATE-OBS",
                primary.get("DATE-OBS"),
            )

            if not date:
                raise ValueError(
                    f"{name}: DATE-OBS is missing."
                )

            observation_time = Time(
                str(date),
                scale="utc",
            )

            wcs = WCS(
                header,
                fobj=hdul,
                naxis=2,
            ).celestial

            if not wcs.has_celestial:
                raise ValueError(
                    f"{name}: no usable celestial WCS."
                )

            image = np.array(
                hdu.data,
                dtype=float,
                copy=True,
            )

            if image.size > 1_000_000:
                raise ValueError(
                    f"{name}: image is too large for this "
                    "inspector. Use a small cutout."
                )

            detector = str(
                header.get(
                    "DETECTOR",
                    primary.get("DETECTOR", "Unknown"),
                )
            )

            frames.append(
                {
                    "name": name,
                    "image": image,
                    "wcs": wcs,
                    "mjd": float(observation_time.mjd),
                    "date": observation_time.isot,
                    "detector": detector,
                }
            )

    frames.sort(key=lambda frame: frame["mjd"])

    reference = frames[0]
    output_wcs = reference["wcs"]
    output_shape = reference["image"].shape

    for frame in frames:
        aligned, footprint = reproject_interp(
            (frame["image"], frame["wcs"]),
            output_wcs,
            shape_out=output_shape,
        )

        valid = (
            (footprint > 0)
            & np.isfinite(aligned)
        )

        aligned[~valid] = np.nan

        if not valid.any():
            raise ValueError(
                f"{frame['name']}: no valid overlap "
                "with the first frame."
            )

        frame["aligned"] = aligned
        frame["coverage"] = float(valid.mean() * 100)

    return [
        {
            key: frame[key]
            for key in (
                "name",
                "aligned",
                "mjd",
                "date",
                "detector",
                "coverage",
            )
        }
        for frame in frames
    ]


def render_sequence(data_folder):
    st.subheader("Aligned sequence inspector")

    st.warning(
        "Exploratory display only. WCS reprojection does not "
        "verify wavelength compatibility, calibration, quality "
        "flags, or a moving object's identity. Visual changes "
        "are not automatically temporal changes."
    )

    paths = sorted(data_folder.glob("*_cutout.fits"))

    if not paths:
        st.info(
            "Download target-centered cutouts first."
        )
        return

    selected = st.multiselect(
        "Choose 2–8 small cutouts of the same sky field",
        options=paths,
        format_func=lambda path: path.name,
        key="sequence_selected_files",
    )

    if not 2 <= len(selected) <= 8:
        st.info("Select between 2 and 8 cutouts.")
        return

    signature = tuple(
        (
            str(path),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in selected
    )

    if st.button(
        "Build aligned sequence",
        key="build_aligned_sequence",
    ):
        st.session_state.pop("aligned_sequence", None)

        try:
            with st.spinner("Reading and aligning cutouts..."):
                items = [
                    (path.name, path.read_bytes())
                    for path in selected
                ]

                frames = build_sequence(items)

            st.session_state["aligned_sequence"] = {
                "signature": signature,
                "frames": frames,
            }

        except Exception as error:
            st.error(
                f"{type(error).__name__}: {error}"
            )

    saved = st.session_state.get("aligned_sequence")

    if saved is None or saved["signature"] != signature:
        return

    frames = saved["frames"]

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "File": frame["name"],
                    "UTC": frame["date"],
                    "Detector": frame["detector"],
                    "Valid output pixels (%)": round(
                        frame["coverage"], 2
                    ),
                }
                for frame in frames
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    span = frames[-1]["mjd"] - frames[0]["mjd"]
    st.caption(
        f"Frames sorted by observation time. "
        f"Time span: {span:.4f} days. "
        "Grid reference: earliest selected frame."
    )

    index = st.slider(
        "Frame",
        min_value=0,
        max_value=len(frames) - 1,
        value=0,
        key="aligned_sequence_frame",
    )

    frame = frames[index]
    image = frame["aligned"]
    finite = image[np.isfinite(image)]

    low, high = np.percentile(finite, [1, 99])

    if high <= low:
        high = low + 1.0

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.imshow(
        np.ma.masked_invalid(image),
        origin="lower",
        cmap="gray",
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )

    ax.set_title(
        f"Frame {index + 1}/{len(frames)} | "
        f"{frame['date']} | "
        f"Detector {frame['detector']}"
    )

    ax.set_xlabel("Reference-grid pixel column")
    ax.set_ylabel("Reference-grid pixel row")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(frame["name"])

    st.caption(
        "Each frame uses its own 1st–99th percentile display "
        "stretch. This is not a quantitative brightness "
        "comparison. Regions outside valid coverage are masked."
    )

    render_animation(frames)


def render_animation(frames):
    from PIL import Image, ImageDraw

    st.subheader("Automatic playback")

    st.caption(
        "Exploratory animation: wavelength compatibility "
        "has not yet been verified. Playback intervals are "
        "uniform, not proportional to elapsed observation time."
    )

    delay_ms = st.select_slider(
        "Time per frame",
        options=[250, 500, 750, 1000, 1500, 2000],
        value=1000,
        format_func=lambda value: f"{value / 1000:g} seconds",
        key="sequence_animation_delay",
    )

    playing = st.checkbox(
        "Play animation",
        value=False,
        key="sequence_animation_play",
    )

    if not playing:
        st.caption(
            "Check Play animation to display the loop. "
            "Uncheck it to hide playback; use the Frame "
            "slider above for individual-image inspection."
        )
        return

    samples = []

    for frame in frames:
        finite = frame["aligned"][
            np.isfinite(frame["aligned"])
        ]

        if finite.size:
            stride = max(1, finite.size // 50000)
            samples.append(finite[::stride])

    if not samples:
        st.error("No finite pixels are available for animation.")
        return

    low, high = np.percentile(
        np.concatenate(samples),
        [1, 99],
    )

    if high <= low:
        high = low + 1.0

    animation_frames = []

    for index, frame in enumerate(frames):
        image = frame["aligned"]
        valid = np.isfinite(image)

        scaled = np.clip(
            (np.nan_to_num(image, nan=low) - low)
            / (high - low),
            0,
            1,
        )

        gray = (scaled * 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)

        # Match the origin="lower" orientation of the inspector.
        rgb = np.flipud(rgb)
        visible = np.flipud(valid)

        # Magenta marks missing coverage, not sky emission.
        rgb[~visible] = [180, 0, 180]

        panel = Image.fromarray(rgb)
        panel.thumbnail((700, 600))

        width = max(520, panel.width)
        canvas = Image.new(
            "RGB",
            (width, panel.height + 80),
            "black",
        )

        canvas.paste(
            panel,
            ((width - panel.width) // 2, 80),
        )

        draw = ImageDraw.Draw(canvas)

        draw.text(
            (10, 8),
            f"Frame {index + 1}/{len(frames)} | "
            f"{frame['date']}",
            fill="white",
        )

        draw.text(
            (10, 28),
            f"Detector {frame['detector']} | "
            "EXPLORATORY - spectral match unverified",
            fill="white",
        )

        draw.text(
            (10, 48),
            "Shared display stretch | Magenta = missing coverage",
            fill="white",
        )

        animation_frames.append(canvas)

    buffer = io.BytesIO()

    animation_frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=animation_frames[1:],
        duration=delay_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )

    gif_bytes = buffer.getvalue()

    st.image(gif_bytes)

    st.download_button(
        "Download exploratory animation",
        data=gif_bytes,
        file_name="skyshift_exploratory_sequence.gif",
        mime="image/gif",
        key="download_exploratory_animation",
    )

    st.caption(
        "A single sampled 1st–99th percentile stretch is used "
        "across the animation. This avoids independently "
        "rescaling each frame, but does not calibrate the "
        "images or make brightness differences scientific."
    )