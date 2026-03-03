# Extension of SVI by J. Gatheral

import matplotlib
from scipy.interpolate import PchipInterpolator
from matplotlib.colors import Normalize
matplotlib.use("Qt5Agg")
import yfinance as yf
import pandas as pd
import numpy as np
from models.svi import SVI
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
plt.style.use('dark_background')

class SSVI(SVI):

