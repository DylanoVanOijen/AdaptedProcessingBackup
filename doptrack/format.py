from typing import List

from doptrack.astro import SatellitePass


def format_list_of_passes(satpasses: List[SatellitePass]) -> str:
    output = [
        f"\n{'':^12}|{'Start of Pass':^14}|{'Max Elevation':^21}|{'End of Pass':^13}",
        f"{'Date':^12}|{'UTC':^7}|{'Az':^6}|{'UTC':^7}|{'Az':^6}|{'El':^6}|{'UTC':^7}|{'Az':^5}",
        '-' * 64
    ]
    for p in satpasses:
        assert p.tca is not None  # TODO Added (temp?) since the predictions in old metafiles don't have tca
        output.append(' | '.join([
            f" {p.time_start.date()}",
            f"{p.time_start.strftime('%H:%M')}",
            f"{p.azimuth_start:>3.0f}°",
            f"{p.tca.strftime('%H:%M')}",
            f"{p.azimuth_tca:>3.0f}°",
            f"{p.max_elevation:>3.0f}°",
            f"{p.time_stop.strftime('%H:%M')}",
            f"{p.azimuth_stop:>3.0f}°"]))
    output.append('-' * 64)
    return '\n'.join(output)
