import logging
from dataclasses import dataclass
from typing import Optional, Iterable, Callable, Union

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fftshift, fftfreq, fft
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.stats import linregress
from sklearn.cluster import DBSCAN

from doptrack.base import DoptrackError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Spectrogram:
    image: np.ndarray
    time: np.ndarray
    frequency: np.ndarray
    dt: float
    df: float
    noise: np.ndarray
    signal_average: np.ndarray
    signal_maximum: np.ndarray
    signal_median: np.ndarray

    def __post_init__(self):
        assert len(self.image.shape)
        assert len(self.time.shape) == 1 and len(self.time) == self.image.shape[0]
        assert len(self.frequency.shape) == 1 and len(self.frequency) == self.image.shape[1]

    def plot(self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs):
        """
        Plot the spectrogram using matplotlib.

        Parameters
        ----------
        ax :
            The axes on which to plot the spectrogram. If no axes are given then a new plot will be made.
        **kwargs :
            Additional settings passed on to imshow.
        """
        extent = (self.frequency[0]/1E6, self.frequency[-1]/1E6, self.time[-1], self.time[0])
        clim = (0.01, 0.2)
        is_standalone_plot = ax is None
        if is_standalone_plot:
            fig, ax = plt.subplots()
        options = dict(cmap="viridis", aspect="auto", extent=extent, clim=clim, interpolation="none")
        options.update(**kwargs)
        img = ax.imshow(10*np.log10(self.image), **options)

        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Time [s]')

        if is_standalone_plot:
            plt.show()
        else:
            return img
        


def create_spectrogram(
        data: Iterable[np.ndarray],
        sampling_rate: int,
        dt: float,
        mask_width: Optional[int] = None,
        mask_center: int = 0
) -> Spectrogram:
    """
    Creates a spectrogram from array data.

    The default spectrogram window width should be wide enough to
    show the whole S-curve for LEO satellites.


    """
    df = 1/dt
    nfft = int(dt * sampling_rate)  # Use optimal value of nfft
    frequency = fftshift(fftfreq(nfft, 1/sampling_rate))
    if mask_width:
        lower = mask_center - mask_width / 2
        upper = mask_center + mask_width / 2
        # Exclude upper limit from mask to give nicer mask length, e.g. 2800 instead of 2801 when including upper limit.
        mask, = np.nonzero((lower <= frequency) & (frequency < upper))
        frequency = frequency[mask]
    else:
        mask = None

    rows = []
    noise = []
    signal_average = []
    signal_maximum = []
    signal_median = []

    for i, chunk in enumerate(data):
        #print("\n", abs(chunk).min(), abs(chunk).max(), abs(chunk).mean())
        row = abs(fftshift(fft(chunk, nfft))) # PSD per chunk
        print(row)
        row = row[mask] if mask is not None else row

        # noise calculation using standard deviation
        mean = np.average(row)
        deviation = np.std(row, dtype=np.float64)
        signal_limit = mean + 2*deviation
        noise_floor = []
        signal_line = []

        for entry in row:
            if entry < mean+deviation and entry > mean-deviation:
                noise_floor.append(entry)
            if entry >= signal_limit:
                signal_line.append(entry)

        noise.append(np.average(noise_floor))
        signal_average.append(np.average(signal_line))
        signal_maximum.append(np.max(signal_line,initial=0))
        signal_median.append(np.median(signal_line))
        #print(row)
        row = row/np.average(noise_floor)
        # noise calculation using only the mean
        #noise.append(np.average(row))
        #row = row/np.average(row)
        rows.append(row)
    image = np.array(rows)
    time = np.arange(image.shape[0]) * dt

     #plt.plot(noise,time)
     #plt.show()
    return Spectrogram(
        image=image,
        time=time,
        frequency=frequency,
        dt=dt,
        df=df,
        noise=noise,
        signal_average=signal_average,
        signal_maximum=signal_maximum,
        signal_median=signal_median,
        )


class SignalNotFound(DoptrackError):
    """Raised when the extraction algorithm is unable to find a satellite signal (S-curve) in the spectrogram."""
    pass


@dataclass
class Signal:
    """A discrete-time signal."""
    time: np.ndarray
    frequency: np.ndarray


@dataclass
class SCurve(Signal):
    """A discrete-time signal representing an S-curve."""
    tca: float
    tca_error: float
    fca: float
    fca_error: float


@dataclass
class Fit:
    model: Callable
    coeffs: np.ndarray
    covar: np.ndarray

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.model(x, *self.coeffs)


def s_curve_model(
        t: float, tca: float, fca: float, a1: float, a2: float, b1: float, b2: float
) -> Union[float, np.ndarray]:
    """A function that fits well to S-curves.

    Neither arctan nor tanh work perfectly by themselves, but the combination
    of both fits almost perfectly to an S-curve. However, the extra degrees of
    freedom result in this model being less robust than a simple arctan or tanh model,
    and it requires a better initial guess and relatively few outliers in order to converge.

    We assume that both functions are zero at tca. This reduces the degrees of freedom
    during the fitting procedure. This should make the fitting slightly more robust,
    and it is a more accurate representation of the physical reality.
    It also makes it trivial to determine both estimated tca and estimated fca from
    the model parameters.
    """
    return a1 * np.arctan((t - tca) / b1) + a2 * np.tanh((t - tca) / b2) + fca


def s_curve_model_robust(x: float, tca: float, fca: float, a: float, b: float) -> Union[float, np.ndarray]:
    """A function that fits robustly to S-curves."""
    return a * np.arctan((x - tca) / b) + fca


def fit_s_curve(time: np.ndarray, frequency: np.ndarray):
    p0 = [400, 7000, -2000, -1000, 100, 100]
    try:
        coeffs, covar = curve_fit(s_curve_model, time, frequency, p0=p0, loss='soft_l1', method='trf')
    except RuntimeError:
        raise SignalNotFound('Non-robust fitting of S-curve failed')
    return Fit(s_curve_model, coeffs, covar)


def fit_robust_s_curve(time: np.ndarray, frequency: np.ndarray):
    p0 = [400, 7000, -2000, 100]
    try:
        coeffs, covar = curve_fit(s_curve_model_robust, time, frequency, p0=p0, loss='soft_l1', method='trf')
    except RuntimeError:
        raise SignalNotFound('Robust fitting of S-curve failed.')
    return Fit(s_curve_model_robust, coeffs, covar)


def extract_s_curve(spectrogram: Spectrogram, sidelobe_distance: int, plot=False) -> SCurve:
    image = spectrogram.image / np.median(spectrogram.image, axis=0)
    dt, df = spectrogram.dt, spectrogram.df

    left_window = create_signal_window(sidelobe_distance=sidelobe_distance, peak=-sidelobe_distance / 2, df=df)
    center_window = create_signal_window(sidelobe_distance=sidelobe_distance, peak=0, df=df)
    right_window = create_signal_window(sidelobe_distance=sidelobe_distance, peak=sidelobe_distance / 2, df=df)

    left_sidelobe = fftconvolve(image, left_window, mode='same')
    center_lobe = fftconvolve(image, center_window, mode='same')
    right_sidelobe = fftconvolve(image, right_window, mode='same')
    convolution = left_sidelobe * center_lobe * right_sidelobe

    signal = get_initial_signal_from_image(convolution, dt=dt, df=df)

    if plot:
        fig, ax = plt.subplots()
        fig.suptitle('Initial data points')
        spectrogram.plot(ax)
        ax.scatter(signal.frequency + spectrogram.frequency[0], signal.time, 5, color='r')
        fig.show()

    # INITIAL FILTERING
    signal = remove_outliers_by_signal_clustering(signal, dt, plot=plot)
    fit = fit_robust_s_curve(signal.time, signal.frequency)
    signal = remove_outliers_by_residual_limiting(signal, fit, limit=sidelobe_distance / 2, plot=plot)

    # INITIAL VALIDATION
    tca, fca = fit.coeffs[0], fit.coeffs[1]
    tca_bounds = (spectrogram.time[0], spectrogram.time[-1])
    if not tca_bounds[0] < tca < tca_bounds[1]:
        raise SignalNotFound(f'The signal has an invalid tca: {tca_bounds[0]} < {tca} < {tca_bounds[1]}')
    fca_bounds = (0, image.shape[1] * df)
    if not fca_bounds[0] < fca < fca_bounds[1]:
        raise SignalNotFound(f'The signal has an invalid fca: {fca_bounds[0]} < {fca} < {fca_bounds[1]}')
    if slope := fit.coeffs[2] > 0:
        raise SignalNotFound(f'The signal fit must have a negative slope: {slope} < 0')

    # FINAL FILTERING
    fit = fit_s_curve(signal.time, signal.frequency)
    signal = remove_outliers_by_residual_clustering(signal, fit, dt=dt, plot=plot)

    # FINAL FITTING
    # Perform a final fit to get the best possible estimate of TCA and FCA
    fit = fit_s_curve(signal.time, signal.frequency)
    tca, fca = fit.coeffs[:2]
    errors = np.sqrt(np.diag(fit.covar))
    tca_error, fca_error = errors[0], errors[1]
    if tca_error > 0.5:
        # If TCA error is too high we will try and use a more robust fit as a last resort
        fit = fit_robust_s_curve(signal.time, signal.frequency)
        tca, fca = fit.coeffs[:2]
        errors = np.sqrt(np.diag(fit.covar))
        tca_error, fca_error = errors[0], errors[1]

    # Move from relative frequency back to absolute frequency.
    # Until this point we use relative frequency since fitting is more robust.
    signal = Signal(signal.time, signal.frequency + spectrogram.frequency[0])
    fca += float(spectrogram.frequency[0])

    logger.debug(f'Estimated TCA: {tca:.3f} ± {tca_error:.3f}')
    logger.debug(f'Estimated FCA: {fca:.3f} ± {fca_error:.3f}')

    # FINAL VALIDATION
    if len(signal.time) < 50:
        raise SignalNotFound('The signal has too few data points')
    if not (signal.time[0] < tca < signal.time[-1]):
        raise SignalNotFound(
            f'The estimated TCA is not within the range of the extracted signal: '
            f'{signal.time[0]} < {tca} < {signal.time[-1]}')
    if tca_error > 0.5:
        # If the TCA error is too high it is a sign of one of two things:
        # 1) An incorrect cluster wasn't filtered out properly.
        # 2) Too few points on one side of TCA.
        # In both cases the data is bad. Testing showed 0.5 to be a good cutoff value.
        raise SignalNotFound(f'The TCA error is too high: {tca_error} > 0.5')

    if plot:
        fig, ax = plt.subplots()
        fig.suptitle('Final extracted S-curve')
        spectrogram.plot(ax)
        ax.scatter(signal.frequency, signal.time, 5, color='r')
        ax.plot(fit(signal.time) + spectrogram.frequency[0], signal.time, color='lime', label='fit')
        ax.legend()
        fig.show()

    return SCurve(
        time=signal.time, frequency=signal.frequency, tca=tca, tca_error=tca_error, fca=fca, fca_error=fca_error
    )


def gaussian(n: np.ndarray, shift: float, std: float) -> np.ndarray:
    return np.exp(-0.5 * ((n - shift) / std) ** 2)


def create_signal_window(sidelobe_distance: int, peak: float, df: float):
    sidelobe_distance_pixels = sidelobe_distance / df
    signal_std_pixels = 5
    peak_pixels = peak / df
    x = np.arange(int(sidelobe_distance_pixels * 1.5 - 1))
    window = gaussian(x, shift=len(x) / 2 + peak_pixels, std=signal_std_pixels)
    return window.reshape((1, len(window)))


def get_initial_signal_from_image(image: np.ndarray, dt: float, df: float) -> Signal:
    pixel_frequency = np.nanargmax(image, axis=1)
    pixel_time = np.arange(image.shape[0])
    non_zero = pixel_frequency != 0
    time = pixel_time[non_zero] * dt
    frequency = pixel_frequency[non_zero] * df
    return Signal(time=time, frequency=frequency)


def remove_outliers_by_signal_clustering(signal: Signal, dt: float, plot=False) -> Signal:
    normalized_frequency = signal.frequency / 15
    data = np.dstack((signal.time, normalized_frequency))[0]
    clustering = DBSCAN(eps=15 * np.sqrt(dt), min_samples=10).fit(data)
    labels = clustering.labels_
    logger.debug(f'Found {len(set(labels))} clusters during signal clustering: {set(labels)}')

    if plot:
        fig, ax = plt.subplots()
        fig.suptitle('Clusters found during signal clustering')
        ax.invert_yaxis()
        for label in set(labels):
            ax.plot(signal.frequency[labels == label], signal.time[labels == label], '.', label=label)
            ax.legend()
        fig.show()

    # We know that the complete S-curve, as well as each individual segment,
    # must have a negative slope. Because of this we take all clusters with
    # positive or near zero slope and designate them as outliers.
    # This removes a lot of smaller invalid clusters and makes the fitting
    # process much more likely to succeed.
    for label in set(labels):
        x, y = signal.time[labels == label], signal.frequency[labels == label]
        slope = linregress(x, y).slope
        if slope > -0.2:
            labels[labels == label] = -1

    outliers = labels == -1
    not_outliers = labels != -1
    if outliers.all():
        raise SignalNotFound('No signal points after signal clustering')
    time, frequency = signal.time[not_outliers], signal.frequency[not_outliers]
    time_outliers, frequency_outliers = signal.time[outliers], signal.frequency[outliers]
    logger.debug(f'Found {len(time)} valid points and {len(time_outliers)} outliers during signal clustering.')

    if plot:
        fig, ax = plt.subplots()
        fig.suptitle('Result of signal clustering and outlier detection')
        ax.invert_yaxis()
        ax.plot(frequency, time, '.', label='valid points')
        ax.plot(frequency_outliers, time_outliers, '.', label='outliers')
        ax.legend()
        fig.show()

    return Signal(time=time, frequency=frequency)


def remove_outliers_by_residual_clustering(signal: Signal, fit: Callable, dt: float, plot=False) -> Signal:
    residual = signal.frequency - fit(signal.time)
    data = np.dstack((signal.time, residual))[0]
    clustering = DBSCAN(eps=20 * np.sqrt(dt), min_samples=10).fit(data)
    labels = clustering.labels_
    logger.debug(f'Found {len(set(labels))} clusters during residual clustering: {set(labels)}')

    if plot:
        fig, ax = plt.subplots()
        fig.suptitle('Clusters found during residual clustering')
        for label in set(labels):
            ax.plot(signal.time[labels == label], residual[labels == label], '.', label=label)
            ax.legend()
        fig.show()

    # If the cluster is located too far from the fitting line we tag it as outliers.
    # Testing showed that a limit value of 10 seemed approriate.
    for label in set(labels):
        cluster = residual[labels == label]
        if abs(np.mean(cluster)) > 10:
            labels[labels == label] = -1

    outliers = labels == -1
    not_outliers = labels != -1
    if outliers.all():
        raise SignalNotFound('No signal points after residual clustering')
    logger.debug(f'Found {sum(not_outliers)} valid points and {sum(outliers)} outliers during residual clustering.')

    if plot:
        fig, ax = plt.subplots()
        fig.suptitle('Result of residual clustering and outlier detection')
        ax.plot(signal.time[not_outliers], residual[not_outliers], '.', label='valid points')
        ax.plot(signal.time[outliers], residual[outliers], '.', label='outliers')
        ax.legend()
        fig.show()

    return Signal(time=signal.time[not_outliers], frequency=signal.frequency[not_outliers])


def remove_outliers_by_residual_limiting(signal: Signal, fit: Callable, limit: float, plot=False) -> Signal:
    residual = signal.frequency - fit(signal.time)
    outliers = abs(residual) > limit
    not_outliers = abs(residual) <= limit
    if outliers.all():
        raise SignalNotFound('No signal points after residual limiting')
    logger.debug(f'Found {sum(not_outliers)} valid points and {sum(outliers)} outliers during residual limiting.')

    if plot:
        fig, ax = plt.subplots()
        fig.suptitle('Result of residual limiting')
        ax.invert_yaxis()
        ax.plot(fit(signal.time), signal.time, 'r', label='fit')
        ax.plot(fit(signal.time) - limit, signal.time, 'y', label='limits')
        ax.plot(fit(signal.time) + limit, signal.time, 'y')
        ax.scatter(signal.frequency[not_outliers], signal.time[not_outliers], 5, 'b', label='valid points')
        ax.scatter(signal.frequency[outliers], signal.time[outliers], 5, 'r', label='outliers')
        ax.legend()
        fig.show()
    return Signal(signal.time[not_outliers], signal.frequency[not_outliers])
