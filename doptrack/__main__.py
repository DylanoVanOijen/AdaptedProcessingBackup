from doptrack.env import setup_logging
from doptrack.cli.main import cli


def main():
    setup_logging()
    cli()


if __name__ == '__main__':
    main()
