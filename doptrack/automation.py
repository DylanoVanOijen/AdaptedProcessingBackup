import csv
import os
from _operator import attrgetter
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Union, Iterator, List, Dict, Set

from doptrack.base import DataID


def create_default_agenda() -> list['ProcessingTask']:
    return [
        ProcessingTask(
            name='Delfi-C3', noradid='32789', signal_frequency=145869000, sidelobe_distance=1200, active=False
        ),
        ProcessingTask(
            name='Delfi-n3Xt', noradid='39428', signal_frequency=145871000, sidelobe_distance=2400, active=False
        ),
        ProcessingTask(
            name='FUNcube', noradid='39444', signal_frequency=145934500, sidelobe_distance=1200, active=False
        ),
        ProcessingTask(
            name='Nayif-1', noradid='42017', signal_frequency=145939000, sidelobe_distance=1200, active=False
        ),
    ]


@dataclass
class ProcessingTask:
    name: str
    noradid: str
    signal_frequency: int
    sidelobe_distance: int
    active: bool


class ProcessingAgenda:
    """A collection of all the processing tasks."""

    def __init__(self, filepath: Union[str, os.PathLike]):
        self.filepath = Path(filepath).expanduser().resolve()

    def get_by_noradid(self, noradid: str) -> ProcessingTask:
        return self._read_file()[noradid]

    def add(self, task: ProcessingTask) -> None:
        agenda = self._read_file()
        if task.noradid in agenda:
            raise ValueError(f'Task with given NORAD id already exists in agenda: {task.noradid}')
        agenda[task.noradid] = task
        self._write_file(agenda)

    def remove(self, noradid: str) -> None:
        agenda = self._read_file()
        if noradid not in agenda:
            raise ValueError(f'Cannot remove non-existing task for NORAD id: {noradid}')
        agenda.pop(noradid)
        self._write_file(agenda)

    def __len__(self) -> int:
        return len(self._read_file())

    def __iter__(self) -> Iterator[ProcessingTask]:
        return iter(self._read_file().values())

    def __contains__(self, noradid: object) -> bool:
        return noradid in self._read_file()

    def __str__(self):
        active: List[ProcessingTask] = [t for t in self if t.active]
        inactive = [t for t in self if not t.active]

        labels = ['SATNAME', 'NORAD ID', 'FREQUENCY', 'SIDELOBE DIST']
        attrs = ['satellite.name', 'satellite.noradid', 'signal_frequency', 'sidelobe_distance']

        # TODO Separate this out to some formatter function
        lengths = []
        rows = [
            labels,
            *[
                [str(attrgetter(attr)(task)) for attr in attrs]
                for task in self
            ]
        ]
        columns = list(map(list, zip(*rows)))
        buffer = 2

        for column in columns:
            lengths.append(max(map(len, column)) + buffer)

        template = "  {:{lengths[0]}} {:>{lengths[1]}} {:>{lengths[2]}} {:>{lengths[3]}}"
        formatter = partial(template.format, lengths=lengths)
        header = formatter(*labels)
        header += '\n  ' + '-' * (len(header) - 2)

        rows = ['\nACTIVE']
        if active:
            rows.append(header)
            for task in active:
                rows.append(formatter(
                    task.name, task.noradid, task.signal_frequency, task.sidelobe_distance
                ))
        else:
            rows.append('  No active tasks!')

        rows.append('\nINACTIVE')
        if inactive:
            rows.append(header)
            for task in inactive:
                rows.append(formatter(
                    task.name, task.noradid, task.signal_frequency, task.sidelobe_distance
                ))
        else:
            rows.append('  No inactive tasks!')
        return '\n'.join(rows) + '\n'

    def _read_file(self) -> Dict[str, ProcessingTask]:
        if not self.filepath.exists():
            return {t.noradid: t for t in create_default_agenda()}
        agenda = {}
        with open(self.filepath, 'r') as file:
            file.readline()  # skip header
            reader = csv.reader(file)
            for row in reader:
                name = row[0]
                noradid = row[1]
                signal_frequency = int(row[2])
                sidelobe_distance = int(row[3])
                active = row[4] == 'True'
                agenda[noradid] = ProcessingTask(
                    name=name,
                    noradid=noradid,
                    signal_frequency=signal_frequency,
                    sidelobe_distance=sidelobe_distance,
                    active=active
                )
        return agenda

    def _write_file(self, agenda: Dict[str, ProcessingTask]):
        with open(self.filepath, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['NAME', 'NORAD ID', 'SIGNAL FREQUENCY', 'SIDELOBE DISTANCE', 'ACTIVE'])
            for task in sorted(agenda.values(), key=lambda e: e.noradid):
                writer.writerow([task.name, task.noradid, task.signal_frequency,
                                 task.sidelobe_distance, task.active])


class ProcessingStatus(Enum):
    SUCCESS = "success"
    NOPASS = "nopass"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ProcessingLogbookEntry:
    dataid: str
    status: ProcessingStatus
    details: str


class ProcessingLogbook:
    """A logbook containing info on whether a dataset succeded or failed during processing."""

    def __init__(self, filepath: Union[str, os.PathLike]):
        self.filepath = Path(filepath).expanduser().resolve()

    def __contains__(self, dataid: str) -> bool:
        return dataid in self.list_all()

    def update(self, dataid: str, status: ProcessingStatus, details: str = '') -> None:
        entries = self._read_file()
        entries[dataid] = ProcessingLogbookEntry(dataid=dataid, status=status, details=details)
        self._write_file(entries)

    def read(self, dataid: str) -> ProcessingLogbookEntry:
        entries = self._read_file()
        if dataid not in entries:
            raise ValueError(f'No processing logbook entry for dataid: {dataid}')
        return entries[dataid]

    def remove(self, dataid: str) -> None:
        entries = self._read_file()
        entries.pop(dataid)
        self._write_file(entries)

    def remove_by_status(self, status: ProcessingStatus) -> None:
        entries = self._read_file()
        entries_to_keep = {dataid: entry for dataid, entry in entries.items() if entry.status != status}
        self._write_file(entries_to_keep)

    def remove_all(self) -> None:
        self._write_file({})

    def list_all(self) -> Set[DataID]:
        entries = self._read_file()
        return {DataID(dataid) for dataid in entries.keys()}

    def list_by_status(self, status: ProcessingStatus) -> Set[DataID]:
        entries = self._read_file()
        return set(DataID(dataid) for dataid, entry in entries.items() if entry.status == status)

    def _read_file(self) -> Dict[str, ProcessingLogbookEntry]:
        if not self.filepath.exists():
            return {}
        entries = {}
        with open(self.filepath, 'r') as file:
            file.readline()  # skip header
            reader = csv.reader(file)
            for row in reader:
                dataid, status, details = row
                entries[dataid] = ProcessingLogbookEntry(
                    dataid=dataid,
                    status=ProcessingStatus(status),
                    details=details,
                )
        return entries

    def _write_file(self, entries: Dict[str, ProcessingLogbookEntry]):
        with open(self.filepath, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['DATAID', 'STATUS', 'DETAILS'])
            for entry in sorted(entries.values(), key=lambda e: e.dataid):
                writer.writerow([entry.dataid, entry.status.value, entry.details])
