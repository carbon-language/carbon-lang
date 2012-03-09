#===- object.py - Python Object Bindings --------------------*- python -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

from ctypes import c_char_p
from ctypes import c_uint64
from ctypes import c_void_p

from .common import LLVMObject
from .common import get_library
from .core import MemoryBuffer

__all__ = [
    "lib",
    "ObjectFile",
    "Relocation",
    "Section",
    "Symbol",
]

class ObjectFile(object):
    """Represents an object/binary file."""

    def __init__(self, filename=None, contents=None):
        """Construct an instance from a filename or binary data.

        filename must be a path to a file that can be opened with open().
        contents can be either a native Python buffer type (like str) or a
        llvm.core.MemoryBuffer instance.
        """
        if contents:
            assert isinstance(contents, MemoryBuffer)

        if filename is not None:
            contents = MemoryBuffer(filename=filename)

        self._memory = contents
        self._obj = lib.LLVMCreateObjectFile(contents)
        contents.release_ownership()
        self._as_parameter_ = self._obj

    def __del__(self):
        lib.LLVMDisposeObjectFile(self)

    def from_param(self):
        return self._as_parameter_

    def get_sections(self):
        """Obtain the sections in this object file.

        This is an iterator for llvm.object.Section instances.
        """
        pass

    def get_symbols(self):
        """Obtain the symbols in this object file.

        This is an iterator for llvm.object.Symbol instances.
        """

class Section(object):
    """Represents a section in an object file."""

    def __init__(self, obj=None):
        """Construct a new section instance.

        Section instances can currently only be created from an ObjectFile
        instance. Therefore, this constructor should not be used outside of
        this module.
        """
        pass

    def __del__(self):
        pass

    @property
    def name(self):
        pass

    @property
    def size(self):
        pass

    @property
    def contents(self):
        pass

    @property
    def address(self):
        pass

    # TODO consider exposing more Pythonic interface, like __contains__
    def has_symbol(self, symbol):
        pass

    def get_relocations(self):
        pass

class Symbol(object):
    def __init__(self):
        pass

    @property
    def name(self):
        pass

    @property
    def address(self):
        pass

    @property
    def file_offset(self):
        pass

    @property
    def size(self):
        pass

class Relocation(object):
    def __init__(self):
        pass

    @property
    def address(self):
        pass

    @property
    def offset(self):
        pass

    @property
    def symbol(self):
        pass

    @property
    def type(self):
        pass

    @property
    def type_name(self):
        pass

    @property
    def value_string(self):
        pass

SectionIteratorRef = c_void_p
SymbolIteratorRef = c_void_p
RelocationIteratorRef = c_void_p

def register_library(library):
    """Register function prototypes with LLVM library instance."""

    # Object.h functions
    library.LLVMCreateObjectFile.argtypes = [MemoryBuffer]
    library.LLVMCreateObjectFile.restype = LLVMObject

    library.LLVMDisposeObjectFile.argtypes = [ObjectFile]

    library.LLVMGetSections.argtypes = [ObjectFile]
    library.LLVMGetSections.restype = SectionIteratorRef

    library.LLVMDisposeSectionIterator.argtypes = [SectionIteratorRef]

    library.LLVMIsSectionIteratorAtEnd.argtypes = [ObjectFile,
            SectionIteratorRef]
    library.LLVMIsSectionIteratorAtEnd.restype = bool

    library.LLVMMoveToNextSection.argtypes = [SectionIteratorRef]

    library.LLVMMoveToContainingSection.argtypes = [SectionIteratorRef,
            SymbolIteratorRef]

    library.LLVMGetSymbols.argtypes = [ObjectFile]
    library.LLVMGetSymbols.restype = SymbolIteratorRef

    library.LLVMDisposeSymbolIterator.argtypes = [SymbolIteratorRef]

    library.LLVMIsSymbolIteratorAtEnd.argtypes = [ObjectFile,
            SymbolIteratorRef]
    library.LLVMIsSymbolIteratorAtEnd.restype = bool

    library.LLVMMoveToNextSymbol.argtypes = [SymbolIteratorRef]

    library.LLVMGetSectionName.argtypes = [SectionIteratorRef]
    library.LLVMGetSectionName.restype = c_char_p

    library.LLVMGetSectionSize.argtypes = [SectionIteratorRef]
    library.LLVMGetSectionSize.restype = c_uint64

    library.LLVMGetSectionContents.argtypes = [SectionIteratorRef]
    library.LLVMGetSectionContents.restype = c_char_p

    library.LLVMGetSectionAddress.argtypes = [SectionIteratorRef]
    library.LLVMGetSectionAddress.restype = c_uint64

    library.LLVMGetSectionContainsSymbol.argtypes = [SectionIteratorRef,
            SymbolIteratorRef]
    library.LLVMGetSectionContainsSymbol.restype = bool

    library.LLVMGetRelocations.argtypes = [SectionIteratorRef]
    library.LLVMGetRelocations.restype = RelocationIteratorRef

    library.LLVMDisposeRelocationIterator.argtypes = [RelocationIteratorRef]

    library.LLVMIsRelocationIteratorAtEnd.argtypes = [SectionIteratorRef,
            RelocationIteratorRef]
    library.LLVMIsRelocationIteratorAtEnd.restype = bool

    library.LLVMMoveToNextRelocation.argtypes = [RelocationIteratorRef]

    library.LLVMGetSymbolName.argtypes = [SymbolIteratorRef]
    library.LLVMGetSymbolName.restype = c_char_p

    library.LLVMGetSymbolAddress.argtypes = [SymbolIteratorRef]
    library.LLVMGetSymbolAddress.restype = c_uint64

    library.LLVMGetSymbolFileOffset.argtypes = [SymbolIteratorRef]
    library.LLVMGetSymbolFileOffset.restype = c_uint64

    library.LLVMGetSymbolSize.argtypes = [SymbolIteratorRef]
    library.LLVMGetSymbolSize.restype = c_uint64

    library.LLVMGetRelocationAddress.argtypes = [SymbolIteratorRef]
    library.LLVMGetRelocationAddress.restype = c_uint64

    library.LLVMGetRelocationOffset.argtypes = [RelocationIteratorRef]
    library.LLVMGetRelocationOffset.restype = c_uint64

    library.LLVMGetRelocationSymbol.argtypes = [RelocationIteratorRef]
    library.LLVMGetRelocationSymbol.restype = SymbolIteratorRef

    library.LLVMGetRelocationType.argtypes = [RelocationIteratorRef]
    library.LLVMGetRelocationType.restype = c_uint64

    library.LLVMGetRelocationTypeName.argtypes = [RelocationIteratorRef]
    library.LLVMGetRelocationTypeName.restype = c_char_p

    library.LLVMGetRelocationValueString.argtypes = [RelocationIteratorRef]
    library.LLVMGetRelocationValueString.restype = c_char_p

lib = get_library()
register_library(lib)
