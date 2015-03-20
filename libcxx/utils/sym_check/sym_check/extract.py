# -*- Python -*- vim: set syntax=python tabstop=4 expandtab cc=80:
"""
extract - A set of function that extract symbol lists from shared libraries.
"""
import distutils.spawn
import sys

from sym_check import util


class Extractor(object):
    """
    Extractor - Extract symbol lists from libraries using nm.
    """

    @staticmethod
    def find_nm():
        """
        Search for the nm executable and return the path and type.
        """
        nm_exe = distutils.spawn.find_executable('nm')
        if nm_exe is not None:
            return nm_exe
        # ERROR no NM found
        print("ERROR: Could not find nm")
        sys.exit(1)

    def __init__(self):
        """
        Initialize the nm executable and flags that will be used to extract
        symbols from shared libraries.
        """
        self.nm_exe = Extractor.find_nm()
        self.flags = ['-P', '-g']

    def extract(self, lib):
        """
        Extract symbols from a library and return the results as a dict of
        parsed symbols.
        """
        cmd = [self.nm_exe] + self.flags + [lib]
        out, _, exit_code = util.execute_command_verbose(cmd)
        if exit_code != 0:
            raise RuntimeError('Failed to run %s on %s' % (self.nm_exe, lib))
        fmt_syms = (self._extract_sym(l)
                    for l in out.splitlines() if l.strip())
            # Cast symbol to string.
        final_syms = (repr(s) for s in fmt_syms if self._want_sym(s))
        # Make unique and sort strings.
        tmp_list = list(sorted(set(final_syms)))
        # Cast string back to symbol.
        return util.read_syms_from_list(tmp_list)

    def _extract_sym(self, sym_str):
        bits = sym_str.split()
        # Everything we want has at least two columns.
        if len(bits) < 2:
            return None
        new_sym = {
            'name': bits[0],
            'type': bits[1]
        }
        new_sym = self._transform_sym_type(new_sym)
        # NM types which we want to save the size for.
        if new_sym['type'] == 'OBJECT' and len(bits) > 3:
            new_sym['size'] = int(bits[3], 16)
        return new_sym

    @staticmethod
    def _want_sym(sym):
        """
        Check that s is a valid symbol that we want to keep.
        """
        if sym is None or len(sym) < 2:
            return False
        bad_types = ['t', 'b', 'r', 'd', 'w']
        return sym['type'] not in bad_types

    @staticmethod
    def _transform_sym_type(sym):
        """
        Map the nm single letter output for type to either FUNC or OBJECT.
        If the type is not recognized it is left unchanged.
        """
        func_types = ['T', 'W']
        obj_types = ['B', 'D', 'R', 'V', 'S']
        if sym['type'] in func_types:
            sym['type'] = 'FUNC'
        elif sym['type'] in obj_types:
            sym['type'] = 'OBJECT'
        return sym


def extract_symbols(lib_file):
    """
    Extract and return a list of symbols extracted from a dynamic library.
    The symbols are extracted using NM. They are then filtered and formated.
    Finally they symbols are made unique.
    """
    extractor = Extractor()
    return extractor.extract(lib_file)
