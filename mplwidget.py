from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide2 import QtWidgets


class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = FigureCanvasQTAgg(Figure())
        # self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QtWidgets.QVBoxLayout(self)
        # layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.axes = self.canvas.figure.add_subplot(111)