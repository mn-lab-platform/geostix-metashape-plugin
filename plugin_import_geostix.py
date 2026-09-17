# Test for dependencies in Metashape Python.
# This install should be executed only once with first run.
from modules.pip_auto_install import pip_install

pip_install("numpy\npandas\nscipy\npillow")

# Main plugin code
import os
from datetime import datetime, timezone
from pathlib import Path

import Metashape
import numpy as np
import pandas as pd
from PIL import ExifTags, Image
from PySide2 import QtWidgets
from PySide2.QtCore import QFile
from PySide2.QtUiTools import QUiLoader
from scipy.optimize import minimize

IMAGE_EXTENSIONS = ("jpg", "jpeg")
GEOSTIX_LOG_COLUMNS = ["TimeOfWeek", "WeekNumber", "FixStatus", "Lat", "Lon", 
                       "Height", "HAccuracy", "VAccuracy", "OrientationX", 
                       "OrientationY", "OrientationZ", "Support1", "Support2", 
                       "Support3", "Support4", "Support5"]
CRS_COMBO_ITEMS = [
    ("WGS 84 (EPSG::4326)", "EPSG::4326"),
    ("CGRS 94 (EPSG::6311)", "EPSG::6311"),
    ("More...", None)
]

UI_FILE_PATH = Path(__file__).parent / "import_geostix.ui"
MENU_ITEM_NAME = "GEOSTIX/Import images and GNSS"

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_UTC_LEAP_SECONDS = 18  # GPS time - UTC, constant since the last leap second (2016-12-31)
SECONDS_PER_WEEK = 7 * 24 * 60 * 60


def get_timestamp_from_exif(path: str) -> float:
    with Image.open(path) as image:
        exif_ifd = image.getexif().get_ifd(ExifTags.IFD.Exif)
    date_string = exif_ifd[ExifTags.Base.DateTimeOriginal]
    subsec_string = exif_ifd.get(ExifTags.Base.SubsecTimeOriginal, "0")
    capture_time = datetime.strptime(date_string, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    subsec = float(f"0.{subsec_string}")
    return (capture_time - GPS_EPOCH).total_seconds() + subsec + GPS_UTC_LEAP_SECONDS

def load_timestamps_from_images(directory: str) -> tuple[pd.DataFrame, list[str]]:
    directory = Path(directory)
    paths = [str(x) for x in directory.iterdir()
             if x.is_file() and x.suffix.lstrip(".").lower() in IMAGE_EXTENSIONS]

    if len(paths) == 0:
        raise RuntimeError("No images found in given directory.")

    timestamps = []
    skipped_paths = []
    for path in paths:
        try:
            timestamps.append(get_timestamp_from_exif(path))
        except Exception:
            skipped_paths.append(path)
            timestamps.append(None)

    dataframe = pd.DataFrame({"path": paths, "timestamp": timestamps}).dropna(subset=["timestamp"])
    if len(dataframe) == 0:
        raise RuntimeError("Failed to read an EXIF timestamp from any image.")
    return dataframe.sort_values("timestamp").reset_index(drop=True), skipped_paths

def load_timestamps_from_log(path: str) -> pd.DataFrame:
    log = pd.read_csv(path, sep=",", names=GEOSTIX_LOG_COLUMNS, index_col=False)
    log["timestamp"] = log["WeekNumber"] * SECONDS_PER_WEEK + log["TimeOfWeek"]
    return log

def find_offset_by_window(first: pd.Series, second: pd.Series, tolerance_seconds: float = 1.0) -> float:
    first_sorted = first.to_numpy()
    second_values = second.to_numpy()
    penalty = 99.0

    middle_value = second_values[len(second_values) // 2]
    candidate_offsets = first_sorted - middle_value

    def score(offset: float) -> float:
        shifted = second_values + offset
        idx = np.searchsorted(first_sorted, shifted)
        idx_left = np.clip(idx - 1, 0, len(first_sorted) - 1)
        idx_right = np.clip(idx, 0, len(first_sorted) - 1)
        left = first_sorted[idx_left]
        right = first_sorted[idx_right]
        nearest = np.where(np.abs(shifted - left) <= np.abs(right - shifted), left, right)
        diffs = np.abs(shifted - nearest)
        return np.sum(np.where(diffs <= tolerance_seconds, diffs ** 2, penalty))

    scores = [score(offset) for offset in candidate_offsets]
    return float(candidate_offsets[np.argmin(scores)])

def match_images_to_log(images: pd.DataFrame, log: pd.DataFrame, tolerance_seconds: float = 1.0) -> pd.DataFrame:
    return pd.merge_asof(
        images.sort_values("timestamp"),
        log.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=tolerance_seconds,
    )

def create_new_chunk_or_default(document: Metashape.Document) -> Metashape.Chunk:
    if len(document.chunks) == 1 and document.chunk.label == "Chunk 1" \
                                    and len(document.chunk.cameras) == 0:
        return document.chunk
    return document.addChunk()

def normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))

def create_new_chunk_from_images(document: Metashape.Document, crs: str, images: pd.DataFrame):
    chunk = create_new_chunk_or_default(document)
    chunk.crs = Metashape.CoordinateSystem(crs)

    chunk.addPhotos(images["path"].tolist())

    # addPhotos does not guarantee chunk.cameras comes back in the same order (or count,
    # e.g. if a photo is already present in the chunk), so match by path instead of position.
    rows_by_path = {normalize_path(row.path): row for row in images.itertuples()}
    for camera in chunk.cameras:
        row = rows_by_path.get(normalize_path(camera.photo.path))
        if row is None:
            continue
        location = np.float64([row.Lon, row.Lat, row.Height])
        location_accuracy = np.float64([row.HAccuracy, row.HAccuracy, row.VAccuracy])
        if np.any(np.isnan(location)):
            continue
        camera.reference.location = Metashape.Vector(location)
        camera.reference.location_accuracy = Metashape.Vector(location_accuracy)

def format_offset(offset_seconds: float) -> str:
    sign = "-" if offset_seconds < 0 else ""
    days, remainder = divmod(abs(offset_seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{sign}{int(days)}d {int(hours)}h {int(minutes)}m {seconds:.3f}s"

def show_summary_dialog(images: pd.DataFrame, offset_seconds: float, skipped_paths: list[str],
                        parent: QtWidgets.QWidget | None = None) -> None:
    total = len(images)
    with_reference = int(images["Lat"].notna().sum())
    message = (
        f"Imported {total} images, {with_reference} with GNSS reference information.\n"
        f"Camera clock offset: {format_offset(offset_seconds)}"
    )
    if skipped_paths:
        names = "\n".join(Path(path).name for path in skipped_paths)
        message += (f"\n\nSkipped {len(skipped_paths)} image(s) with no readable "
                    f"EXIF timestamp:\n{names}")
    QtWidgets.QMessageBox.information(parent, "Import summary", message)

def path_if_exist(path: str) -> str:
    if not os.path.exists(path):
        raise FileExistsError(f"Path {path} do not exist.")
    return path
    
def execute_import(document: Metashape.Document, dialog: QtWidgets.QDialog) -> None:
    try:
        image_directory = path_if_exist(dialog.image_directory_text.text())
        log_file_path = path_if_exist(dialog.log_file_text.text())
        crs = dialog.crs_combo.currentData()
        images, skipped_paths = load_timestamps_from_images(image_directory)
        log = load_timestamps_from_log(log_file_path)
        offset = find_offset_by_window(log["timestamp"], images["timestamp"])

        images_for_import = images.copy()
        images_for_import["timestamp"] = images_for_import["timestamp"] + offset
        images_for_import = match_images_to_log(images_for_import, log)
        create_new_chunk_from_images(document, crs, images_for_import)
        show_summary_dialog(images_for_import, offset, skipped_paths, dialog)
    except Exception as error:
        QtWidgets.QMessageBox.critical(dialog, "Import failed", str(error))

def browse_for_image_directory(dialog: QtWidgets.QDialog) -> None:
    directory = QtWidgets.QFileDialog.getExistingDirectory(
        dialog, "Select image directory", dialog.image_directory_text.text()
    )
    if directory:
        dialog.image_directory_text.setText(directory)

def browse_for_log_file(dialog: QtWidgets.QDialog) -> None:
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        dialog, "Select log file", dialog.log_file_text.text()
    )
    if path:
        dialog.log_file_text.setText(path)

def prompt_for_custom_crs(combo: QtWidgets.QComboBox, more_index: int, previous_index: int) -> None:
    crs = Metashape.app.getCoordinateSystem("Select Coordinate System")
    combo.blockSignals(True)
    try:
        if crs is None:
            combo.setCurrentIndex(previous_index)
        else:
            combo.insertItem(more_index, f"{crs.name} ({crs.authority})", crs.authority)
            combo.setCurrentIndex(more_index)
    finally:
        combo.blockSignals(False)

def setup_crs_combo(dialog: QtWidgets.QDialog) -> None:
    combo = dialog.crs_combo
    for text, data in CRS_COMBO_ITEMS:
        combo.addItem(text, data)
    state = {"previous_index": combo.currentIndex()}

    def on_current_index_changed(index: int) -> None:
        if combo.itemData(index) is not None:
            state["previous_index"] = index
            return
        prompt_for_custom_crs(combo, index, state["previous_index"])
        state["previous_index"] = combo.currentIndex()

    combo.currentIndexChanged.connect(on_current_index_changed)

def load_dialog(parent: QtWidgets.QWidget | None = None) -> QtWidgets.QDialog:
    loader = QUiLoader()
    ui_file = QFile(str(UI_FILE_PATH))
    ui_file.open(QFile.OpenModeFlag.ReadOnly)
    dialog = loader.load(ui_file, parent)
    ui_file.close()
    return dialog

def import_geostix_tool():
    document = Metashape.app.document
    qapplication = QtWidgets.QApplication.instance()
    parent = qapplication.activeWindow()
    dialog = load_dialog(parent=parent)
    button_box = dialog.button_box
    ok_button = button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    ok_button.clicked.connect(lambda: execute_import(document, dialog))
    dialog.image_directory_button.clicked.connect(lambda: browse_for_image_directory(dialog))
    dialog.log_file_button.clicked.connect(lambda: browse_for_log_file(dialog))
    setup_crs_combo(dialog)
    dialog.exec()
