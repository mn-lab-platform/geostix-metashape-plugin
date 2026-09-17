from dataclasses import dataclass
from pathlib import Path

import Metashape
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm
from PySide2 import QtWidgets
from PySide2.QtCore import QFile
from PySide2.QtUiTools import QUiLoader

from mplwidget import MplWidget

PLUGIN_UI_FILE_PATH = Path(__file__).parent / "align.ui"
PLOT_UI_FILE_PATH = Path(__file__).parent / "plot.ui"
MENU_ITEM_NAME = "GEOSTIX/Align and analyse"
ACCURACY_OPTIONS = [
    ("Highest", 0),
    ("High", 1),
    ("Medium", 2),
    ("Low", 4),
    ("Lowest", 8)
]

@dataclass
class Statistics:
    number_of_projections: np.ndarray
    unit_error_vectors: np.ndarray
    unit_errors: np.ndarray
    metric_error_vectors: np.ndarray
    metric_errors: np.ndarray
    unit_error_3d: np.ndarray
    unit_error: float

def set_cameras_reference_enabled(chunk: Metashape.Chunk, value: bool):
    for camera in chunk.cameras:
        camera.reference.enabled = value

def chunk_has_aligned_cameras(chunk: Metashape.Chunk) -> bool:
    return any(camera.transform for camera in chunk.cameras)

def run_alignment(chunk: Metashape.Chunk, accuracy: int, generic_preselection: bool, reference_preselection: bool):
    chunk.matchPhotos(downscale=accuracy, generic_preselection=generic_preselection, 
                          reference_preselection=reference_preselection)
    chunk.alignCameras(reset_alignment=True)

def setup_initial_lever(chunk: Metashape.Chunk, lever_arm: Metashape.Vector):
    sensor = chunk.sensors[0]
    sensor.axes = Metashape.Sensor.Axes.Terrestrial
    sensor.antenna.location_ref = lever_arm
    sensor.antenna.location_fixed = True

def unlock_lever(sensor: Metashape.Sensor):
    sensor.antenna.location_fixed = False

def update_transform(chunk: Metashape.Chunk):
    chunk.updateTransform()

def optimize(chunk: Metashape.Chunk):
    chunk.optimizeCameras()

def align_without_reference(chunk: Metashape.Chunk, accuracy: int, 
                            generic_preselection: bool, reference_preselection: bool,
                            lever_arm: Metashape.Vector):
    set_cameras_reference_enabled(chunk, False)
    run_alignment(chunk, accuracy, generic_preselection, reference_preselection)
    setup_initial_lever(chunk, lever_arm)
    set_cameras_reference_enabled(chunk, True)
    update_transform(chunk)

def optimize_with_reference(chunk: Metashape.Chunk):
    unlock_lever(chunk.sensors[0])
    optimize(chunk)

def get_antenna_transform(sensor):
    location = sensor.antenna.location
    if location is None:
        location = sensor.antenna.location_ref
    if location is None:
        location = Metashape.Vector([0.0, 0.0, 0.0])
    rotation = sensor.antenna.rotation
    if rotation is None:
        rotation = sensor.antenna.rotation_ref
    if rotation is None:
        rotation = Metashape.Vector([0.0, 0.0, 0.0])
    terrestial = Metashape.Matrix([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    return terrestial * Metashape.Matrix.Translation(location) * Metashape.Matrix.Rotation(Metashape.Utils.ypr2mat(rotation))

def accuracy_to_weight(a: np.ndarray) -> np.ndarray:
    aa = a * a
    p = [1 / aa[0], 1 / aa[1], 1 / aa[2]]
    return np.float64(p)

def get_estimated_position(camera: Metashape.Camera) -> Metashape.Vector:
    camera_transform = camera.chunk.transform.matrix * camera.transform
    antenna_transform = get_antenna_transform(camera.sensor)
    estimated_geoccs = camera_transform.translation() + camera_transform.rotation() * antenna_transform.translation()
    return estimated_geoccs

def get_reference_position(camera: Metashape.Camera) -> Metashape.Vector:
    reference_crs = camera.reference.location
    reference_geoccs = Metashape.CoordinateSystem.transform(reference_crs, camera.chunk.crs, camera.chunk.crs.geoccs)
    return reference_geoccs

def get_location_weight(camera: Metashape.Camera) -> np.ndarray:
    acc = camera.reference.location_accuracy
    weight = np.float64([1 / acc[0], 1 / acc[1], 1 / acc[2]])
    return weight, weight * weight

def count_projections_for_camera(camera: Metashape.Camera) -> int:
    return len(camera.chunk.tie_points.projections[camera])

def compute_statistics(chunk: Metashape.Chunk) -> Statistics:
    crs = chunk.crs
    geoccs = crs.geoccs

    number_of_projections = []
    metric_error_vectors = []
    metric_errors = []
    unit_error_vectors = []
    unit_errors = []
    unit_error_3d = np.zeros(3)

    cameras = [x for x in chunk.cameras if x.reference.enabled and x.transform]
    num_cameras = len(cameras)
    if num_cameras < 2:
        raise RuntimeError(
            f"Not enough aligned cameras with a GNSS reference to compute statistics "
            f"(found {num_cameras}, need at least 2)."
        )
    for camera in cameras:
        position_estimated_geoccs = get_estimated_position(camera)
        position_reference_geoccs = get_reference_position(camera)
        localframe = geoccs.localframe(position_estimated_geoccs)
        position_estimated_local = localframe.mulp(position_estimated_geoccs)
        position_reference_local = localframe.mulp(position_reference_geoccs)

        error_vector = np.asarray(position_estimated_local - position_reference_local)
        error_sq_vector = error_vector * error_vector
        weight, weight_sq = get_location_weight(camera)
        unit_error_vector = weight * error_vector
        
        metric_error_vectors.append(error_vector)
        metric_errors.append(np.linalg.norm(error_vector))
        unit_error_vectors.append(unit_error_vector)
        unit_errors.append(np.linalg.norm(weight @ error_vector))
        unit_error_3d += weight_sq * error_sq_vector

        camera_projections = count_projections_for_camera(camera)
        number_of_projections.append(camera_projections)

    return Statistics(
        np.int64(number_of_projections),
        np.float64(unit_error_vectors),
        np.float64(unit_errors),
        np.float64(metric_error_vectors),
        np.float64(metric_errors),
        np.sqrt( unit_error_3d / (num_cameras - 1)),
        np.sqrt( unit_error_3d.sum() / (num_cameras - 1))
    )
    
def format_statistics_label(statistics: Statistics) -> str:
    unit_error_3d = ", ".join(f"{value:.4f}" for value in statistics.unit_error_3d)
    return f"m0 = {statistics.unit_error:.4f} ({unit_error_3d})"

def create_polar_histogram(plot: MplWidget, errors: np.ndarray, angle_bins: int = 12, 
                            magnitude_bins: int = 3, cmap: str = "viridis"):
    figure = plot.canvas.figure
    figure.clear()
    axes = figure.add_subplot(111, projection="polar")
    magnitudes = np.linalg.norm(errors, axis=1)
    angles = np.arctan2(errors[:, 1], errors[:, 0]) % (2 * np.pi)

    angle_edges = np.linspace(0, 2 * np.pi, angle_bins + 1)
    magnitude_edges = np.linspace(0, magnitudes.max(), magnitude_bins + 1)
    counts, _, _ = np.histogram2d(angles, magnitudes, bins=[angle_edges, magnitude_edges])
    counts_masked = np.ma.masked_where(counts == 0, counts)

    max_count = int(counts_masked.max())
    discrete_cmap = colormaps[cmap].resampled(max_count)
    norm = BoundaryNorm(np.arange(0.5, max_count + 1.5), discrete_cmap.N)

    mesh = axes.pcolormesh(angle_edges, magnitude_edges, counts_masked.T, cmap=discrete_cmap, norm=norm)
    figure.colorbar(mesh, ax=axes, pad=0.1, ticks=np.arange(1, max_count + 1))
    plot.canvas.draw()

def create_regular_histogram(plot: MplWidget, values: np.ndarray):
    figure = plot.canvas.figure
    figure.clear()
    axes = figure.add_subplot(111)
    axes.hist(values)
    plot.canvas.draw()

def plot_statistics(dialog, prefix: str, statistics: Statistics) -> None:
    ANGLE_BINS = 24
    MAGNITUDE_BINS = 5

    create_polar_histogram(getattr(dialog, f"{prefix}_mxy_plot"), statistics.metric_error_vectors[:, [0, 1]],
                           angle_bins=ANGLE_BINS, magnitude_bins=MAGNITUDE_BINS)
    create_polar_histogram(getattr(dialog, f"{prefix}_mxz_plot"), statistics.metric_error_vectors[:, [0, 2]],
                               angle_bins=ANGLE_BINS, magnitude_bins=MAGNITUDE_BINS)
    create_polar_histogram(getattr(dialog, f"{prefix}_myz_plot"), statistics.metric_error_vectors[:, [1, 2]],
                               angle_bins=ANGLE_BINS, magnitude_bins=MAGNITUDE_BINS)
    create_regular_histogram(getattr(dialog, f"{prefix}_unit_plot"), statistics.unit_errors)        

def display_analysis_window(parent: QtWidgets.QWidget, initial_statistics: Statistics,
                            optimize_statistics: Statistics) -> None:
    loader = QUiLoader()
    loader.registerCustomWidget(MplWidget)
    ui_file = QFile(str(PLOT_UI_FILE_PATH))
    ui_file.open(QFile.OpenModeFlag.ReadOnly)
    dialog = loader.load(ui_file, parent)
    ui_file.close()

    dialog.initial_label.setText(format_statistics_label(initial_statistics))
    plot_statistics(dialog, "initial", initial_statistics)

    dialog.optimized_label.setText(format_statistics_label(optimize_statistics))
    plot_statistics(dialog, "optimized", optimize_statistics)

    dialog.exec()

def execute(parent, dialog, chunk: Metashape.Chunk):
    if chunk_has_aligned_cameras(chunk):
        answer = QtWidgets.QMessageBox.question(
            parent, "Chunk already aligned",
            "This chunk already has aligned cameras. Continuing will realign them and "
            "discard the existing alignment. Do you want to proceed?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            print("Align and analyse cancelled: chunk already has aligned cameras.")
            return

    try:
        lever_arm = Metashape.Vector([dialog.lever_x_spin.value(),
                                      dialog.lever_y_spin.value(),
                                      dialog.lever_z_spin.value()])

        align_without_reference(chunk, dialog.accuracy_combo.currentData(),
                                dialog.generic_preselection_check.isChecked(),
                                dialog.reference_preselection_check.isChecked(),
                                lever_arm)
        initial_statistics = compute_statistics(chunk)
        optimize_with_reference(chunk)
        optimize_statistics = compute_statistics(chunk)
        display_analysis_window(parent, initial_statistics, optimize_statistics)
    except Exception as error:
        QtWidgets.QMessageBox.critical(parent, "Align and analyse failed", str(error))


def load_align_dialog(parent, chunk: Metashape.Chunk) -> QtWidgets.QDialog:
    loader = QUiLoader()
    ui_file = QFile(str(PLUGIN_UI_FILE_PATH))
    ui_file.open(QFile.OpenModeFlag.ReadOnly)
    dialog = loader.load(ui_file, parent)
    ui_file.close()

    for label, data in ACCURACY_OPTIONS:
        dialog.accuracy_combo.addItem(label, data)
    dialog.accuracy_combo.setCurrentIndex(1)
    dialog.lever_x_spin.setValue(0.0)
    dialog.lever_y_spin.setValue(-chunk.sensors[0].focal_length / 1000.0)
    dialog.lever_z_spin.setValue(0.205)
    return dialog

def align_and_analyse():
    document = Metashape.app.document
    chunk = document.chunk
    qapplication = QtWidgets.QApplication.instance()
    parent = qapplication.activeWindow()

    if chunk is None or len(chunk.cameras) == 0 or len(chunk.sensors) == 0:
        QtWidgets.QMessageBox.warning(parent, "No photos",
                                      "The active chunk has no photos to align. Add photos first.")
        return

    dialog = load_align_dialog(parent, chunk)
    dialog.button_box.accepted.connect(lambda: execute(parent, dialog, chunk))
    dialog.exec()