from __future__ import annotations

import logging
from typing import BinaryIO, Union
from io import BytesIO, StringIO
from csv import reader
import re

import numpy as np
import pandas as pd

import fsspec
import fsspec.core
from fsspec.core import get_fs_token_paths
import fsspec.utils

lggr = logging.getLogger("csv-to-partitions")

def make_df(
    n_cols: int,
    rows_str: str,
    sep: str,
    names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read a CSV into a DataFrame without NaN value conversion so that any None values are
    only present due to a missing field, permitting a check for parsed CSV column count.
    """
    if names is None:
        names = [str(n) for n in range(n_cols)]
    return pd.read_csv(
        StringIO(rows_str),
        names=names,
        sep=sep,
        keep_default_na=False,
        na_filter=False,
        na_values=[],
        engine="python",
        converters=dict.fromkeys(range(n_cols), trivial_return),
    )

def trivial_return(value):
    "Return a value unchanged (used to override pandas CSV parser dtype conversion)."
    return value

def validate_df(
    df: pd.DataFrame,
    escapechar: str = r"\\",
    quotechar: str = '"',
    doublequote: bool = True,
):
    """
    Validate that the DataFrame does not contain ``None`` values and does not have
    ``quotechar`` characters within any field values. Arguments other than those below
    are the same as those passed to :function:`pandas.read_csv` (``escapechar``,
    ``quotechar``, ``doublequote``).

    Args:
      df      : (:class:`pd.DataFrame`) DataFrame which can be iterated to give rows to
                be validated in the form of :class:`pd.Series` objects (one per row).
    """
    n_matchable = 1 + int(doublequote)
    re_patt_str = (
        rf".*((?<!({escapechar}))({quotechar}))" + r"{1," + f"{n_matchable}" + r"}.*"
    )
    re_patt = re.compile(re_patt_str)
    for row_idx, row_series in df.iterrows():
        row = row_series.to_dict()
        if None in row.values():
            raise ValueError(f"Absent field (incomplete row) at {row=}")
        if any(isinstance(v, str) and re_patt.match(v) for v in row.values()):
            raise ValueError(f"{quotechar=} found in {row=}")
    return

class SingleCsvToPartitions:
    """Translate the content of one CSV file into CSV partition metadata.

    The 'tail' of each partition (except the final one) is read in reverse, so as to
    validate the row terminator (which could otherwise be a newline character within a
    multiline quoted string field value, continued from a preceding row). The 'tail' is
    the sequence of bytes ending at the first row terminator after the ``offset``, and
    beginning ``sample_rows`` rows (default: 10) earlier.


    The row terminator is identified greedily as the first ``lineseparator`` before
    which the ``sample_rows`` rows can be successfully parsed. The attempt to parse the
    tail is carried out in reverse (from the row terminator backwards) so that parsing
    will 'fail fast' from unsuccessful row terminating newlines. When a newline fails to
    parse correctly as a row terminator, then the validation is repeated for the
    subsequent ``linseparator``.

    Parameters
    ----------
    csv : file-like
        Input CSV file as a binary Python file-like object (duck-typed, adhering
        to BinaryIO is optional)
    url : string
        URI of the CSV file.
    n_columns : int
        The number of columns
    sep : str
        Delimiter to use when reading the CSV.
    offsets : list[int]
        The list of offsets into the file at which to seek to place CSV partitions.
    lineseparator : str
        The newline character to match as a candidate row terminator.
    spec : int
        The version of output to produce (see README of this repo)
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g. host,
        port, username, password, etc.  (optional, default ``None``).
    sample_tail_rows : int
        The number of rows to sample from the tail. Zero or negative to disable
    """

    def __init__(
        self,
        csv: BinaryIO,
        url: str,
        n_columns: int,
        sep: str = ",",
        blocksize: int = 2 ** 25,
        lineseparator: str = "\n",
        escapechar: str = "\\",
        quotechar: str = '"',
        spec: int = 1,
        storage_options: dict | None = None,
        sample_tail_rows: int = 10,
    ):
        # Open CSV file in read mode...
        lggr.debug(f"CSV file: {csv}")
        self.input_file = csv

        fs, fs_token, paths = get_fs_token_paths(url, mode="rb", storage_options=storage_options)
        path = paths[0] # simplified case: presume only handling a single file
        self._size = fs.info(path)["size"]
        lggr.debug(f"Total size: {self._size}")
        self.n_columns = n_columns
        self.sep = sep
        self.blocksize = blocksize
        offsets = list(range(0, self._size, blocksize))
        lggr.debug(f"{offsets=}")

        self.unadjusted_offsets = offsets

        self.linesep = lineseparator
        self.escapechar = escapechar
        self.quotechar = quotechar
        self.spec = spec
        self.storage_options = storage_options or {}
        self.sample_tail_rows = sample_tail_rows
        self.non_row_term_offsets = []
        # For the purpose of testing
        # use the CSV handle again here? maybe tail sample buffer?
        # maybe dict of tail sample buffers for each offset?
        #self._csv = csv

        self.store = {}

        self._uri = url
        lggr.debug(f"CSV file URI: {self._uri}")

    def translate(self):
        """Translate content of one CSV file into partition offsets format.

        This method is the main entry point to execute the Dask workflow, and
        returns a "reference" structure to be used with `dask.read_bytes`

        Only ``sample_tail_rows`` worth of bytes are copied out of the CSV file, for
        each partition end point in ``offsets``.

        Returns
        -------
        dict
            Dictionary containing reference structure.
        """
        lggr.debug("Translation begins")
        for o in self.unadjusted_offsets:
            self._translator(o)
        if self.spec < 1:
            return self.store
        else:
            for k, v in self.store.copy().items():
                self.store[k][0] = "{{u}}"
            return {"version": 1, "templates": {"u": self._uri}, "refs": self.store}

    def _translator(self, offset: int):
        """Produce offset metadata for all partitions in the CSV file."""
        lggr.debug(f"Translating CSV dataset at {offset}")
        # Store chunk location metadata...
        offset_idx = self.unadjusted_offsets.index(offset)
        if offset:
            prev_start = next(k for k in reversed(self.store) if k < offset)
            prev_len = self.store[prev_start][2]
            start_offset = prev_start + prev_len
            # Return early if cannot translate offset due to EOF or blocksize overshoot
            if start_offset >= offset + self.blocksize:
                lggr.debug(
                    f"Skipping {offset} from {start_offset} and {self.blocksize} "
                    f"-- {start_offset} is not in the range [{offset},{offset+self.blocksize})"
                )
                return
            elif start_offset == self._size:
                lggr.debug(f"Skipping {offset} from {start_offset} -- end of file reached")
                return
        else:
            start_offset = 0
        lggr.debug(f"Handling {offset} as {start_offset}")
        if offset == self.unadjusted_offsets[-1]:
            # The final offset will go up to the end of the file. There is no next one
            # to check backwards from, so just use the end of the file as the offset.
            next_row_term_offset = self._size
        else:
            next_unadjusted_offset = self.unadjusted_offsets[offset_idx + 1]
            next_row_term_offset = self._rowterminatorat(next_unadjusted_offset)
        size = next_row_term_offset - start_offset
        chunk_key = start_offset
        if size:
            self.store[chunk_key] = [self._uri, start_offset, size]

    def _rowterminatorat(self, offset: int) -> int:
        """
        GOAL: Find the first row-terminating lineseparator after the given offset, by
        reading backwards up to that offset.

        IMPLEMENTED: Find the first row-terminating lineseparator after the given
        offset, by matching the first newline character.
        """
        revlinebuf = BytesIO()
        rows_to_sample = self.sample_tail_rows
        skip_linesep_count = 0
        # seek to line terminator from offset
        # increment skip_linesep_count if it's not a valid linesep
        t = offset - 1
        self.input_file.seek(t)
        char = ""
        while offset > 0 and char != self.linesep:
            char = self.input_file.read(1)
            t += 1
            if t in self.non_row_term_offsets:
                pass # Won't re-check, already recorded as non-row-terminating
            elif char == self.linesep:
                if not self.linesep_is_row_terminating(t):
                    self.non_row_term_offsets.append(t)
            else:
                continue
            if t in self.non_row_term_offsets:
                # Skip any already encountered non-row-terminating newline
                t += 1
                # Overwrite the newline char so the while loop doesn't exit
                char = self.input_file.read(1)
        lggr.debug(f"Transformed {offset} to {t}")
        return t

    def linesep_is_row_terminating(self, offset: int):
        """
        Determine if the given position is the location of a row-terminating newline, or
        merely a newline within a multiline field, by attempting to parse as many rows
        as specified in the :attr:`sample_tail_rows` attribute (default: 10) and either
        encountering a parser error (to indicate the newline is not row-terminating)
        or reaching the start of the file (which would indicate the newline is too early
        in the file to read all the requested sample tail rows, but is otherwise valid).
        
        Args:
          offset : The position of the newline in the file
        """
        starting_tell = self.input_file.tell() # To be restored afterwards
        line_count = 0
        rows_str = ""
        # Step back 2 so that when you read forward 1 you don't get the newline
        t = offset - 2
        self.input_file.seek(t)
        char = ""
        while t >= 0 and line_count < self.sample_tail_rows:
            char = self.input_file.read(1)
            if char == self.linesep:
                # The offsets are stored with an off-by-one error,
                # indicating the first character after the newline,
                # rather than the position of the [non-row-terminating] newline
                if (t + 1) not in self.non_row_term_offsets:
                    # Do not increment if the 
                    line_count += 1
            rows_str += char
            t -= 1 # Step back a character to read the next
            if t >= 0:
                self.input_file.seek(t)
            else:
                # If the start of the file is reached, there's an implicit newline
                line_count += 1
        # Now make a df using ``rows_str`` string of ``sample_tail_rows`` many rows
        if not rows_str.endswith(self.linesep):
            lggr.debug(f"Adding a linesep to {rows_str!r}")
            rows_str += self.linesep
        try:
            rows_df = make_df(n_cols=self.n_columns, rows_str=rows_str, sep=self.sep)
        except:
            # Any ParserError indicates it's not a valid row-terminating offset
            is_row_terminating = False
        else:
            try:
                validate_df(rows_df)
            except:
                # Validation error indicates it's not a valid row-terminating offset
                is_row_terminating = False
            else:
                print(f"Decided {rows_str!r} is row-terminating at {offset}")
                is_row_terminating = True
        if not is_row_terminating:
            print(f"Decided {rows_str!r} is non-row-terminating at {offset}")
        ## Simulate (testing and) identifying a row terminating newline
        #if t == 26:
        #    is_row_terminating = False
        #else:
        #    is_row_terminating = True
        self.input_file.seek(starting_tell)
        return is_row_terminating
