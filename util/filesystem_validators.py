import argparse
import os
from pathlib import Path


class ReadableFile(argparse.Action):

    def __call__(self, parser, parser_namespace, values: Path, option_string=None):
        if not values.exists():
            raise argparse.ArgumentError(self, f"'{values.absolute()}' does not exist")

        if not values.is_file():
            raise argparse.ArgumentError(self, f"'{values.absolute()}' must be a file")

        if not os.access(str(values.absolute()), os.R_OK):
            raise argparse.ArgumentError(self, f"'{values.absolute()}' must be readable")

        setattr(parser_namespace, self.dest, values)


class WriteableFile(argparse.Action):

    def __call__(self, parser, parser_namespace, values: Path, option_string=None):
        if not values.is_file():
            raise argparse.ArgumentError(self, f"'{values.absolute()}' must be a file")

        if not values.exists():
            values.mkdir()

        elif not os.access(str(values.absolute()), os.W_OK):
            raise argparse.ArgumentError(self, f"'{values.absolute()}' must be writable")

        setattr(parser_namespace, self.dest, values)


class WriteableDirectory(argparse.Action):

    def __call__(self, parser, parser_namespace, values: Path, option_string=None):
        if not values.is_dir():
            raise argparse.ArgumentError(self, f"'{values.absolute()}' must be a directory")

        if not values.exists():
            values.mkdir()

        elif not os.access(str(values.absolute()), os.W_OK):
            raise argparse.ArgumentError(self, f"'{values.absolute()}' must be writeable")

        setattr(parser_namespace, self.dest, values)
