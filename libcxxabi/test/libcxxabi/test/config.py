import locale
import os
import platform
import re
import shlex
import sys

import lit.Test  # pylint: disable=import-error,no-name-in-module
import lit.util  # pylint: disable=import-error,no-name-in-module

from libcxx.test.format import LibcxxTestFormat
from libcxx.test.config import Configuration as LibcxxConfiguration
from libcxx.compiler import CXXCompiler

class Configuration(LibcxxConfiguration):
    # pylint: disable=redefined-outer-name
    def __init__(self, lit_config, config):
        super(Configuration, self).__init__(lit_config, config)

    def configure_src_root(self):
        self.libcxxabi_src_root = self.get_lit_conf('libcxxabi_src_root',
            os.path.dirname(self.config.test_source_root))
        self.libcxx_src_root = self.get_lit_conf('libcxx_src_root',
            os.path.join(self.libcxxabi_src_root, '/../libcxx'))

    def configure_obj_root(self):
        self.obj_root = self.get_lit_conf('libcxxabi_obj_root',
                                          self.libcxxabi_src_root)

    def configure_compile_flags(self):
        self.cxx.compile_flags += ['-DLIBCXXABI_NO_TIMER']
        super(Configuration, self).configure_compile_flags()

    def configure_compile_flags_header_includes(self):
        cxx_headers = self.get_lit_conf('cxx_headers',
            os.path.join(self.libcxx_src_root, '/include'))
        if not os.path.isdir(cxx_headers):
            self.lit_config.fatal("cxx_headers='%s' is not a directory."
                                  % cxx_headers)
        self.cxx.compile_flags += ['-I' + cxx_headers]

        libcxxabi_headers = self.get_lit_conf('libcxxabi_headers',
                                              os.path.join(self.libcxxabi_src_root,
                                                           'include'))
        if not os.path.isdir(libcxxabi_headers):
            self.lit_config.fatal("libcxxabi_headers='%s' is not a directory."
                                  % libcxxabi_headers)
        self.cxx.compile_flags += ['-I' + libcxxabi_headers]

    def configure_compile_flags_exceptions(self):
        pass

    def configure_compile_flags_rtti(self):
        pass

    def configure_compile_flags_no_threads(self):
        self.cxx.compile_flags += ['-DLIBCXXABI_HAS_NO_THREADS=1']

    def configure_compile_flags_no_monotonic_clock(self):
        pass

    def configure_link_flags_abi_library_path(self):
        # Configure ABI library paths.
        self.cxx.link_flags += ['-L' + self.obj_root,
                                '-Wl,-rpath,' + self.obj_root]

    def configure_env(self):
        if sys.platform == 'darwin':
            self.env['DYLD_LIBRARY_PATH'] = self.obj_root
