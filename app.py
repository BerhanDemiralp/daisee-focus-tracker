from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from mvp_focus import FaceCropper, FocusModel, RollingFocusMonitor, TURKEY_TZ


DEFAULT_CHECKPOINT = Path("models/best_model.pt")
REPORT_DIR = Path("reports")


st.set_page_config(page_title="Odak Takibi MVP", layout="wide")


@st.cache_resource
def get_focus_model(checkpoint_path: str) -> FocusModel:
    return FocusModel(checkpoint_path)


@st.cache_resource
def get_face_cropper() -> FaceCropper:
    return FaceCropper()


def init_state() -> None:
    if "monitor" not in st.session_state:
        st.session_state.monitor = RollingFocusMonitor()
    if "prediction_rows" not in st.session_state:
        st.session_state.prediction_rows = []
    if "last_image" not in st.session_state:
        st.session_state.last_image = None
    if "live_camera_access_confirmed" not in st.session_state:
        st.session_state.live_camera_access_confirmed = False
    if "live_camera_stop_requested" not in st.session_state:
        st.session_state.live_camera_stop_requested = False
    if "live_camera_running" not in st.session_state:
        st.session_state.live_camera_running = False
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Tek goruntu"


def reset_monitor(window_size: int, min_window_size: int, warning_threshold: float, soft_warning_threshold: float) -> None:
    st.session_state.monitor = RollingFocusMonitor(
        window_size=window_size,
        min_window_size=min_window_size,
        warning_threshold=warning_threshold,
        soft_warning_threshold=soft_warning_threshold,
    )
    st.session_state.prediction_rows = []
    st.session_state.last_image = None
    st.session_state.live_camera_access_confirmed = False
    st.session_state.live_camera_stop_requested = True
    st.session_state.live_camera_running = False


def predict_image(image: Image.Image, focus_model: FocusModel, use_face_crop: bool) -> dict:
    timestamp = datetime.now(TURKEY_TZ)
    model_image = image.convert("RGB")

    if use_face_crop:
        face = get_face_cropper().crop(model_image)
        if face is None:
            current_scores = st.session_state.monitor.scores
            row = {
                "zaman": timestamp.strftime("%H:%M:%S"),
                "tahmin": "Yuz Bulunamadi",
                "odak_olasiligi": 0.0,
                "anlik_odak_skoru": 0.0,
                "ortalama_odak_skoru": float(sum(current_scores) / len(current_scores)) if current_scores else 0.0,
                "durum": "Yuz Bulunamadi",
                "uyari": "",
                "yuz_bulundu": False,
            }
            st.session_state.prediction_rows.append(row)
            return row
        model_image = face

    label, focused_probability, frame_focus_score = focus_model.predict(model_image)
    window_focus_score, status, alert = st.session_state.monitor.update(frame_focus_score, timestamp)
    row = {
        "zaman": timestamp.strftime("%H:%M:%S"),
        "tahmin": "Yuksek Odak" if label == "focused" else "Dusuk Odak",
        "odak_olasiligi": focused_probability,
        "anlik_odak_skoru": frame_focus_score,
        "ortalama_odak_skoru": window_focus_score,
        "durum": status,
        "uyari": alert,
        "yuz_bulundu": True,
    }
    st.session_state.prediction_rows.append(row)
    st.session_state.last_image = model_image
    return row


def open_camera() -> cv2.VideoCapture | None:
    camera_candidates = [
        (0, cv2.CAP_DSHOW),
        (0, cv2.CAP_MSMF),
        (0, cv2.CAP_ANY),
        (1, cv2.CAP_DSHOW),
        (1, cv2.CAP_ANY),
    ]

    for camera_index, backend in camera_candidates:
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)

        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            time.sleep(0.1)

        cap.release()

    return None


def render_current_status(row: dict | None) -> None:
    if row is None:
        st.info("Henuz tahmin yok.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Anlik odak skoru", f"{row['anlik_odak_skoru']:.1f}")
    col2.metric("Ortalama odak skoru", f"{row['ortalama_odak_skoru']:.1f}")
    col3.metric("Durum", row["durum"])


def render_history() -> None:
    rows = st.session_state.prediction_rows
    if not rows:
        st.info("Odak gecmisi henuz bos.")
        return

    df = pd.DataFrame(rows)
    st.line_chart(
        df[["anlik_odak_skoru", "ortalama_odak_skoru"]].rename(
            columns={
                "anlik_odak_skoru": "Anlik odak skoru",
                "ortalama_odak_skoru": "Ortalama odak skoru",
            }
        )
    )
    st.dataframe(df.tail(30), use_container_width=True)

    st.subheader("Durum gecmisi")
    alerts = [row["uyari"] for row in rows if row.get("uyari")]
    if alerts:
        for alert in alerts[-10:]:
            st.info(alert)
    else:
        st.info("Henuz durum bildirimi yok.")


init_state()

st.title("Odak Takibi MVP")
st.caption("Upload edilen gorseller dogrudan modele verilir. Kamera snapshot'lari once yuz algilama ile crop edilir.")

with st.sidebar:
    st.header("Model")
    checkpoint_path = st.text_input("Model dosyasi", value=str(DEFAULT_CHECKPOINT))
    if not Path(checkpoint_path).exists():
        st.error("Model dosyasi bulunamadi")
        st.stop()
    focus_model = get_focus_model(checkpoint_path)

    st.header("Rolling window")
    window_size = st.slider("Ortalama pencere boyutu", min_value=3, max_value=30, value=10)
    min_window_size = st.slider("Uyari icin minimum goruntu", min_value=1, max_value=20, value=5)
    warning_threshold = st.slider("Dusuk odak esigi", min_value=0, max_value=100, value=45)
    soft_warning_threshold = st.slider("Ortalama odak esigi", min_value=0, max_value=100, value=53)

    if st.button("Gecmisi sifirla"):
        reset_monitor(
            window_size=window_size,
            min_window_size=min_window_size,
            warning_threshold=float(warning_threshold),
            soft_warning_threshold=float(soft_warning_threshold),
        )
        st.rerun()

    show_processed_image = st.checkbox("Islenen yuz goruntusunu goster", value=True)

page = st.radio(
    "Sayfa",
    ["Tek goruntu", "Canli kamera", "Odak gecmisi"],
    horizontal=True,
    label_visibility="collapsed",
)

if st.session_state.active_page != page:
    if st.session_state.active_page == "Canli kamera":
        st.session_state.live_camera_stop_requested = True
        st.session_state.live_camera_running = False
    st.session_state.active_page = page

if page == "Tek goruntu":
    st.subheader("Tek goruntu ile tahmin")

    uploaded_files = st.file_uploader(
        "Goruntu yukle",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    latest_row = st.session_state.prediction_rows[-1] if st.session_state.prediction_rows else None
    if uploaded_files:
        for uploaded_file in uploaded_files:
            latest_row = predict_image(Image.open(uploaded_file), focus_model, use_face_crop=False)
        st.success(f"{len(uploaded_files)} goruntu islendi.")

    camera_col, score_col = st.columns([1, 1])
    with camera_col:
        st.markdown("#### Kamera snapshot")
        show_snapshot_camera = st.checkbox("Tek goruntu kamerasi ac", value=False)
        if show_snapshot_camera:
            camera_image = st.camera_input("Kameradan tek snapshot al")
            if camera_image is not None:
                latest_row = predict_image(Image.open(camera_image), focus_model, use_face_crop=True)
                st.success("Kamera snapshot yuz algilama ile islendi.")
        else:
            st.info("Canli kamera ile cakismamasi icin snapshot kamerasi kapali.")

    with score_col:
        st.markdown("#### Skorlar")
        render_current_status(latest_row)
        if show_processed_image and st.session_state.last_image is not None:
            st.image(st.session_state.last_image, caption="Modele verilen yuz goruntusu", width=260)

elif page == "Canli kamera":
    st.subheader("Her saniye snapshot alan canli kamera")

    duration_seconds = st.slider("Calisma suresi (saniye)", min_value=5, max_value=300, value=60, step=5)
    st.session_state.live_camera_access_confirmed = st.checkbox(
        "Kamera iznini verdim, canli akisi baslat",
        value=st.session_state.live_camera_access_confirmed,
    )

    live_placeholder = st.empty()
    live_chart = st.empty()
    live_preview = st.empty()

    start_col, stop_col = st.columns([1, 1])
    with start_col:
        start_live = st.button(
            "Canli snapshot akisini baslat",
            type="primary",
            disabled=not st.session_state.live_camera_access_confirmed or st.session_state.live_camera_running,
        )
    with stop_col:
        stop_live = st.button(
            "Canli akisi durdur / kamerayi kapat",
        )

    if stop_live:
        st.session_state.live_camera_stop_requested = True
        st.session_state.live_camera_running = False
        st.info("Canli akis durduruluyor, kamera kapatilacak.")

    if start_live:
        st.session_state.live_camera_stop_requested = False
        st.session_state.live_camera_running = True
        cap = open_camera()
        if cap is None:
            st.session_state.live_camera_running = False
            st.error(
                "Kameradan goruntu alinamadi. Kamerayi kullanan baska uygulamalari kapatip "
                "tarayiciyi yenileyin. Harici kamera kullaniyorsaniz Windows kamera ayarlarini kontrol edin."
            )
        else:
            progress = st.progress(0)
            try:
                for step in range(int(duration_seconds)):
                    if st.session_state.live_camera_stop_requested or st.session_state.active_page != "Canli kamera":
                        st.info("Canli akis durduruldu.")
                        break

                    ok, frame_bgr = cap.read()
                    if not ok:
                        st.error("Kameradan goruntu alinamadi.")
                        break

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    row = predict_image(Image.fromarray(frame_rgb), focus_model, use_face_crop=True)

                    live_placeholder.metric(
                        label=row["durum"],
                        value=f"{row['ortalama_odak_skoru']:.1f}",
                        delta=f"Anlik skor: {row['anlik_odak_skoru']:.1f}",
                    )

                    if show_processed_image and st.session_state.last_image is not None:
                        live_preview.image(
                            st.session_state.last_image,
                            caption="Modele verilen yuz goruntusu",
                            width=260,
                        )

                    df = pd.DataFrame(st.session_state.prediction_rows)
                    live_chart.line_chart(
                        df[["anlik_odak_skoru", "ortalama_odak_skoru"]].rename(
                            columns={
                                "anlik_odak_skoru": "Anlik odak skoru",
                                "ortalama_odak_skoru": "Ortalama odak skoru",
                            }
                        )
                    )

                    progress.progress((step + 1) / int(duration_seconds))
                    time.sleep(1.0)
            finally:
                cap.release()
                st.session_state.live_camera_running = False

            output_path = REPORT_DIR / "live_camera" / "app_live_predictions.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(st.session_state.prediction_rows).to_csv(output_path, index=False)
            st.success(f"Canli kamera sonuclari kaydedildi: {output_path}")

elif page == "Odak gecmisi":
    render_history()
