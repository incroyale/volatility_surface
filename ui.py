import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from surface import volSurface
import numpy as np

plt.style.use('dark_background')


class volSurfaceUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volatility Surface - Bloomberg Style")
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        self.vs = None  # Store the volSurface instance
        self.iv_df = None
        self.current_strike = None
        self.current_term = None
        self.init_data()
        self.init_ui()

    def init_data(self):
        self.vs = volSurface(ticker="^SPX")
        print("Fetching option chain...")
        self.vs.fetch_option_chain()
        self.iv_df = self.vs.fetch_iv_df(strike_width=0.15, max_days=0.5, min_days=0)
        print(f"IV data rows: {len(self.iv_df)}")
        print(f"IV data head:\n{self.iv_df.head()}")

        # Get available strikes and terms
        self.available_strikes = self.vs.get_available_strikes()
        self.available_terms = self.vs.get_available_terms()

        print(f"Available strikes: {len(self.available_strikes)} strikes")
        print(f"Strike range: {min(self.available_strikes) if self.available_strikes else 'None'} to {max(self.available_strikes) if self.available_strikes else 'None'}")
        print(f"Available terms: {len(self.available_terms)} terms")
        print(f"Term range: {min(self.available_terms) if self.available_terms else 'None'} to {max(self.available_terms) if self.available_terms else 'None'}")

        # Defaults
        if self.available_strikes and self.available_terms:
            self.current_strike = self.available_strikes[len(self.available_strikes) // 2]
            self.current_term = self.available_terms[0]
            print("Data loaded successfully")
        else:
            print("ERROR: No strikes or terms available!")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel - Surface plot
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        self.surface_figure = Figure(figsize=(8, 6), facecolor='#1e1e1e')
        self.surface_canvas = FigureCanvas(self.surface_figure)
        left_layout.addWidget(self.surface_canvas)

        # Right panel - Smile and Term Structure
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # Smile panel
        smile_panel = QWidget()
        smile_layout = QVBoxLayout(smile_panel)
        smile_controls = QHBoxLayout()
        smile_label = QLabel("Select Term:")
        smile_label.setStyleSheet("color: #00d4ff; font-weight: bold; font-size: 12px;")
        self.term_dropdown = QComboBox()
        self.term_dropdown.setStyleSheet("background-color: #2d2d2d; color: #ffffff; padding: 8px; font-size: 12px; min-width: 120px;")
        self.term_dropdown.addItems([f"{int(t * 365)} days" for t in self.available_terms])
        self.term_dropdown.currentIndexChanged.connect(self.update_smile)
        smile_controls.addWidget(smile_label)
        smile_controls.addWidget(self.term_dropdown)
        smile_controls.addStretch()
        smile_layout.addLayout(smile_controls)
        self.smile_figure = Figure(figsize=(6, 3), facecolor='#1e1e1e')
        self.smile_canvas = FigureCanvas(self.smile_figure)
        smile_layout.addWidget(self.smile_canvas)
        right_layout.addWidget(smile_panel)

        # Term structure panel
        term_panel = QWidget()
        term_layout = QVBoxLayout(term_panel)
        term_controls = QHBoxLayout()
        term_label = QLabel("Select Strike:")
        term_label.setStyleSheet("color: #00d4ff; font-weight: bold; font-size: 12px;")
        self.strike_dropdown = QComboBox()
        self.strike_dropdown.setStyleSheet("background-color: #2d2d2d; color: #ffffff; padding: 8px; font-size: 12px; min-width: 120px;")
        self.strike_dropdown.addItems([f"{s:.0f}" for s in self.available_strikes])
        self.strike_dropdown.setCurrentIndex(len(self.available_strikes) // 2)
        self.strike_dropdown.currentIndexChanged.connect(self.update_term_structure)
        term_controls.addWidget(term_label)
        term_controls.addWidget(self.strike_dropdown)
        term_controls.addStretch()
        term_layout.addLayout(term_controls)
        self.term_figure = Figure(figsize=(6, 3), facecolor='#1e1e1e')
        self.term_canvas = FigureCanvas(self.term_figure)
        term_layout.addWidget(self.term_canvas)
        right_layout.addWidget(term_panel)

        # Add panels to main layout
        main_layout.addWidget(left_panel, 60)
        main_layout.addWidget(right_panel, 40)

        # Initial plots
        self.plot_surface()
        self.update_smile(0)
        self.update_term_structure(len(self.available_strikes) // 2)

    def plot_surface(self):
        self.surface_figure.clear()
        ax = self.surface_figure.add_subplot(111, projection='3d')
        ax.set_facecolor('#1e1e1e')

        # Build surface using instance method
        strike_grid, T_grid, iv_surface = self.vs.build_iv_surface()
        surf = ax.plot_surface(strike_grid, T_grid, iv_surface, cmap='plasma', alpha=1, edgecolor='none', antialiased=True)
        ax.set_xlabel('Strike', color='#00d4ff', fontsize=10)
        ax.set_ylabel('Time to Maturity', color='#00d4ff', fontsize=10)
        ax.set_zlabel('Implied Volatility', color='#00d4ff', fontsize=10)
        ax.set_title('Implied Volatility Surface', color='#ffffff', fontsize=12, pad=20)
        ax.tick_params(colors='#ffffff', labelsize=8)
        ax.view_init(elev=25, azim=225)
        self.surface_canvas.draw()


    def update_smile(self, index):
        self.current_term = self.available_terms[index]
        self.smile_figure.clear()
        ax = self.smile_figure.add_subplot(111)
        ax.set_facecolor('#1e1e1e')

        # Interpolate smile
        atm_iv, interpolated_df, original_df = self.vs.interpolate_smile(self.current_term)
        spot = self.vs.fetch_spot_price()
        ax.scatter(original_df["strike"], original_df["impliedVolatility"], color='#00d4ff', s=30, label='Market Data', zorder=3)
        ax.plot(interpolated_df["strike"], interpolated_df["impliedVolatility"], color='#00d4ff', linewidth=2)
        ax.axvline(spot, color='#ff6b35', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Strike', color='#00d4ff', fontsize=9)
        ax.set_ylabel('Implied Volatility', color='#00d4ff', fontsize=9)
        ax.set_title(f'Volatility Smile (T={int(self.current_term * 365)} days)', color='#ffffff', fontsize=10)
        ax.tick_params(colors='#ffffff', labelsize=8)
        ax.grid(True, alpha=0.2, color='#555555')
        self.smile_canvas.draw()

    def update_term_structure(self, index):
        self.current_strike = self.available_strikes[index]
        self.term_figure.clear()
        ax = self.term_figure.add_subplot(111)
        ax.set_facecolor('#1e1e1e')

        # Get Term Structure
        term_df = self.vs.get_term_structure(self.current_strike)
        ax.plot(term_df["T"], term_df["impliedVolatility"],marker='o', color='#00d4ff', linewidth=2, markersize=5)
        ax.set_xlabel('Time to Maturity', color='#00d4ff', fontsize=9)
        ax.set_ylabel('Implied Volatility', color='#00d4ff', fontsize=9)
        ax.set_title(f'Term Structure (Strike={self.current_strike:.0f})', color='#ffffff', fontsize=10)
        ax.tick_params(colors='#ffffff', labelsize=8)
        ax.grid(True, alpha=0.2, color='#555555')
        self.term_canvas.draw()


def main():
    app = QApplication(sys.argv)
    window = volSurfaceUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()