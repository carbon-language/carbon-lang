#===- common.py - Python LLVM Bindings -----------------------*- python -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

from ctypes import cdll

import ctypes.util
import platform

__all__ = [
    "find_library",
    "get_library",
]

def find_library():
    # FIXME should probably have build system define absolute path of shared
    # library at install time.
    for lib in ["LLVM-3.1svn", "LLVM"]:
        result = ctypes.util.find_library(lib)
        if result:
            return result

    # FIXME This is a local hack to ease development.
    return "/usr/local/llvm/lib/libLLVM-3.1svn.so"

def get_library():
    """Obtain a reference to the llvm library."""
    lib = find_library()
    if not lib:
        raise Exception("LLVM shared library not found!")

    return cdll.LoadLibrary(lib)
