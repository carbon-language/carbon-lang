#===- core.py - Python LLVM Bindings -------------------------*- python -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

from .common import get_library

from ctypes import POINTER
from ctypes import byref
from ctypes import c_char_p
from ctypes import c_void_p

__all__ = [
    "lib",
    "MemoryBufferRef",
]

lib = get_library()

class MemoryBuffer(object):
    """Represents an opaque memory buffer."""

    def __init__(self, filename=None):
        """Create a new memory buffer.

        Currently, we support creating from the contents of a file at the
        specified filename.
        """
        if filename is None:
            raise Exception("filename argument must be defined")

        memory = c_void_p(None)
        out = c_char_p(None)

        result = lib.LLVMCreateMemoryBufferWithContentsOfFile(filename,
                byref(memory), byref(out))

        if result:
            raise Exception("Could not create memory buffer: %s" % out.value)

        self._memory = memory

    def __del__(self):
        lib.LLVMDisposeMemoryBuffer(self._memory)

    def from_param(self):
        return self._memory


def register_library(library):
    library.LLVMCreateMemoryBufferWithContentsOfFile.argtypes = [c_char_p,
            POINTER(c_void_p), POINTER(c_char_p)]
    library.LLVMCreateMemoryBufferWithContentsOfFile.restype = bool

    library.LLVMDisposeMemoryBuffer.argtypes = [c_void_p]

register_library(lib)
