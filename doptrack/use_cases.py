import gc
import logging

from matplotlib import figure

from doptrack.base import DoptrackError, ApplicationException, DataID
from doptrack.processing import create_spectrogram_product, TrackingProduct, create_tracking_product, \
    ValidationProduct, \
    create_validation_product, SpectrogramProduct
from doptrack.signals import SignalNotFound, extract_s_curve
from doptrack.automation import ProcessingLogbook, ProcessingStatus, ProcessingAgenda
from doptrack.recording import Recording
from doptrack.repositories import DataNotFoundError, DataRepository, ProcessingRepository

logger = logging.getLogger(__name__)


def delete_processed_data(
        logbook: ProcessingLogbook,
        processing_repository: ProcessingRepository,
        datatype_bound: str
) -> None:
    if processing_repository:
        if datatype_bound == 'spectrogram':
            processing_repository.spectrogram.delete_all()
            processing_repository.tracking.delete_all()
            processing_repository.validation.delete_all()
        elif datatype_bound == 'processing':
            processing_repository.tracking.delete_all()
            processing_repository.validation.delete_all()
        elif datatype_bound == 'validation':
            processing_repository.validation.delete_all()
        else:
            raise ApplicationException(f'Unknown data type: {datatype_bound}')
        processing_repository.summary.delete_all()
        logbook.remove_by_status(ProcessingStatus.SUCCESS)
    else:
        raise ApplicationException('Unable to delete processed data without a processing repository.')


def process_recordings(
        agenda: ProcessingAgenda,
        logbook: ProcessingLogbook,
        recording_repository: DataRepository[Recording],
        processing_repository: ProcessingRepository,
        failed=False,
        nopass=False
) -> None:
    if failed:
        logbook.remove_by_status(ProcessingStatus.ERROR)
        logbook.remove_by_status(ProcessingStatus.WARNING)
    if nopass:
        logbook.remove_by_status(ProcessingStatus.NOPASS)

    all_dataids = set.union(recording_repository.list(), processing_repository.list())

    noradids_to_process = {task.noradid for task in agenda if task.active}
    dataids_with_task = {dataid for dataid in all_dataids if dataid.noradid in noradids_to_process}
    dataids_already_processed = logbook.list_all()
    dataids_to_process = dataids_with_task.difference(dataids_already_processed)

    if dataids_to_process:
        logger.info(f'Starting processing of {len(dataids_to_process)} datasets')
        for dataid in sorted(dataids_to_process):
            _process_recording(
                dataid=dataid,
                agenda=agenda,
                logbook=logbook,
                recording_repository=recording_repository,
                processing_repository=processing_repository,
            )
            gc.collect()  # Manual garbage collection invoked to keep memory usage as low as possible
        logger.info('Finished processing all datasets')
    else:
        logger.info('All datasets already processed')


def _process_recording(
        dataid: DataID,
        agenda: ProcessingAgenda,
        logbook: ProcessingLogbook,
        recording_repository: DataRepository[Recording],
        processing_repository: ProcessingRepository,
) -> None:
    """
    Note: Don't call any .list() methods on repositories in this function.
    When the repositories have thousands or even hundreds of thousands of
    files the .list() methods can take several seconds. This can lead to huge
    slowdowns when processing thousands of datasets.
    Assume that whoever called this function has checked that the data is present
    in the necessary repositories.
    """

    task = agenda.get_by_noradid(dataid.noradid)

    def process_spectrogram() -> SpectrogramProduct:
        recording = recording_repository.load(dataid)
        spectrogram = create_spectrogram_product(recording, center_frequency=task.signal_frequency, dt=0.5)
        processing_repository.spectrogram.save(dataid, spectrogram)
        return spectrogram

    def process_tracking() -> TrackingProduct:
        try:
            spectrogram = processing_repository.spectrogram.load(dataid)
        except DataNotFoundError:
            spectrogram = process_spectrogram()
        tracking = create_tracking_product(spectrogram, sidelobe_distance=task.sidelobe_distance)
        processing_repository.tracking.save(dataid, tracking)
        return tracking

    def process_validation() -> ValidationProduct:
        try:
            tracking = processing_repository.tracking.load(dataid)
        except DataNotFoundError:
            tracking = process_tracking()
        validation = create_validation_product(tracking, station=tracking.meta.station, satellite=tracking.meta.satellite)
        processing_repository.validation.save(dataid, validation)
        return validation

    def plot_nopass() -> None:
        spectrogram = processing_repository.spectrogram.load(dataid)
        if spectrogram is not None:
            fig = figure.Figure(figsize=(15, 12))
            ax = fig.subplots()
            fig.suptitle(dataid)
            spectrogram.plot(ax)
            processing_repository.plotting.save(f"nopass/{dataid}", fig)

    def plot_success() -> None:
        try:
            spectrogram = processing_repository.spectrogram.load(dataid)
        except DataNotFoundError:
            spectrogram = None

        validation = processing_repository.validation.load(dataid)

        if spectrogram:
            fig = figure.Figure(figsize=(15, 12))
            ax = fig.subplots()
            fig.suptitle(dataid)
            spectrogram.plot(ax)
            ax.scatter(validation.frequency, validation.time, 1, "r")
            processing_repository.plotting.save(f"extraction/{dataid}", fig)

        fig = figure.Figure(figsize=(15, 12))
        axs = fig.subplots(nrows=3, ncols=1)
        fig.suptitle(dataid)
        axs[0].plot(validation.time, validation.rangerate)
        axs[0].plot(validation.time, validation.rangerate_tle)
        axs[1].plot(validation.time, validation.first_residual)
        axs[2].plot(validation.time, validation.second_residual)
        processing_repository.plotting.save(f"residuals/{dataid}", fig)

    logger.info(f'Starting processing of dataset: {dataid}')
    try:
        data = process_validation()
    except SignalNotFound as e:
        plot_nopass()
        logbook.update(dataid, status=ProcessingStatus.NOPASS, details=repr(e))
        logger.info(f'No satellite pass found during processing of dataset: {dataid}')
    except DoptrackError as e:
        logbook.update(dataid, status=ProcessingStatus.WARNING, details=repr(e))
        logger.warning(f'Processing of dataset failed ({e}): {dataid}')
    except Exception as e:
        logbook.update(dataid, status=ProcessingStatus.ERROR, details=repr(e))
        logger.exception(f'Processing of dataset failed unexpectedly: {dataid}')
    else:
        plot_success()
        processing_repository.summary.add(dataid, data)
        logbook.update(dataid, status=ProcessingStatus.SUCCESS)
        logger.info(f'Processing successful: {dataid}')


def inspect_extraction(dataid: str, agenda: ProcessingAgenda, processing_repository: ProcessingRepository):
    spectrogram = processing_repository.spectrogram.load(dataid)
    task = agenda.get_by_noradid(spectrogram.meta.satellite.noradid)
    extract_s_curve(spectrogram, sidelobe_distance=task.sidelobe_distance, plot=True)
