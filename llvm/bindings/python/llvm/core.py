#===- core.py - Python LLVM Bindings -------------------------*- python -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

from .common import LLVMObject
from .common import c_object_p
from .common import get_library

from . import enumerations

from ctypes import POINTER
from ctypes import byref
from ctypes import c_char_p
from ctypes import c_uint

__all__ = [
    "lib",
    "OpCode",
    "MemoryBuffer",
    "Module",
    "Context",
    "PassRegistry"
]

lib = get_library()

class OpCode(object):
    """Represents an individual OpCode enumeration."""

    _value_map = {}

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return 'OpCode.%s' % self.name

    @staticmethod
    def from_value(value):
        """Obtain an OpCode instance from a numeric value."""
        result = OpCode._value_map.get(value, None)

        if result is None:
            raise ValueError('Unknown OpCode: %d' % value)

        return result

    @staticmethod
    def register(name, value):
        """Registers a new OpCode enumeration.

        This is called by this module for each enumeration defined in
        enumerations. You should not need to call this outside this module.
        """
        if value in OpCode._value_map:
            raise ValueError('OpCode value already registered: %d' % value)

        opcode = OpCode(name, value)
        OpCode._value_map[value] = opcode
        setattr(OpCode, name, opcode)

class MemoryBuffer(LLVMObject):
    """Represents an opaque memory buffer."""

    def __init__(self, filename=None):
        """Create a new memory buffer.

        Currently, we support creating from the contents of a file at the
        specified filename.
        """
        if filename is None:
            raise Exception("filename argument must be defined")

        memory = c_object_p()
        out = c_char_p(None)

        result = lib.LLVMCreateMemoryBufferWithContentsOfFile(filename,
                byref(memory), byref(out))

        if result:
            raise Exception("Could not create memory buffer: %s" % out.value)

        LLVMObject.__init__(self, memory, disposer=lib.LLVMDisposeMemoryBuffer)

    def __len__(self):
        return lib.LLVMGetBufferSize(self)

class Module(LLVMObject):
    """Represents the top-level structure of an llvm program in an opaque object."""

    def __init__(self, module, name=None, context=None):
        LLVMObject.__init__(self, module, disposer=lib.LLVMDisposeModule)

    @classmethod
    def CreateWithName(cls, module_id):
        m = Module(lib.LLVMModuleCreateWithName(module_id))
        c = Context.GetGlobalContext().take_ownership(m)
        return m

    @property
    def datalayout(self):
        return lib.LLVMGetDataLayout(self)

    @datalayout.setter
    def datalayout(self, new_data_layout):
        """new_data_layout is a string."""
        lib.LLVMSetDataLayout(self, new_data_layout)

    @property
    def target(self):
        return lib.LLVMGetTarget(self)

    @target.setter
    def target(self, new_target):
        """new_target is a string."""
        lib.LLVMSetTarget(self, new_target)

    def dump(self):
        lib.LLVMDumpModule(self)

    def print_module_to_file(self, filename):
        out = c_char_p(None)
        # Result is inverted so 0 means everything was ok.
        result = lib.LLVMPrintModuleToFile(self, filename, byref(out))        
        if result:
            raise RuntimeError("LLVM Error: %s" % out.value)

class Context(LLVMObject):

    def __init__(self, context=None):
        if context is None:
            context = lib.LLVMContextCreate()
            LLVMObject.__init__(self, context, disposer=lib.LLVMContextDispose)
        else:
            LLVMObject.__init__(self, context)

    @classmethod
    def GetGlobalContext(cls):
        return Context(lib.LLVMGetGlobalContext())

class PassRegistry(LLVMObject):
    """Represents an opaque pass registry object."""

    def __init__(self):
        LLVMObject.__init__(self,
                            lib.LLVMGetGlobalPassRegistry())

def register_library(library):
    # Initialization/Shutdown declarations.
    library.LLVMInitializeCore.argtypes = [PassRegistry]
    library.LLVMInitializeCore.restype = None

    library.LLVMInitializeTransformUtils.argtypes = [PassRegistry]
    library.LLVMInitializeTransformUtils.restype = None

    library.LLVMInitializeScalarOpts.argtypes = [PassRegistry]
    library.LLVMInitializeScalarOpts.restype = None

    library.LLVMInitializeObjCARCOpts.argtypes = [PassRegistry]
    library.LLVMInitializeObjCARCOpts.restype = None

    library.LLVMInitializeVectorization.argtypes = [PassRegistry]
    library.LLVMInitializeVectorization.restype = None

    library.LLVMInitializeInstCombine.argtypes = [PassRegistry]
    library.LLVMInitializeInstCombine.restype = None

    library.LLVMInitializeIPO.argtypes = [PassRegistry]
    library.LLVMInitializeIPO.restype = None

    library.LLVMInitializeInstrumentation.argtypes = [PassRegistry]
    library.LLVMInitializeInstrumentation.restype = None

    library.LLVMInitializeAnalysis.argtypes = [PassRegistry]
    library.LLVMInitializeAnalysis.restype = None

    library.LLVMInitializeIPA.argtypes = [PassRegistry]
    library.LLVMInitializeIPA.restype = None

    library.LLVMInitializeCodeGen.argtypes = [PassRegistry]
    library.LLVMInitializeCodeGen.restype = None

    library.LLVMInitializeTarget.argtypes = [PassRegistry]
    library.LLVMInitializeTarget.restype = None

    library.LLVMShutdown.argtypes = []
    library.LLVMShutdown.restype = None

    # Pass Registry declarations.
    library.LLVMGetGlobalPassRegistry.argtypes = []
    library.LLVMGetGlobalPassRegistry.restype = c_object_p

    # Context declarations.
    library.LLVMContextCreate.argtypes = []
    library.LLVMContextCreate.restype = c_object_p

    library.LLVMContextDispose.argtypes = [Context]
    library.LLVMContextDispose.restype = None

    library.LLVMGetGlobalContext.argtypes = []
    library.LLVMGetGlobalContext.restype = c_object_p

    # Memory buffer declarations
    library.LLVMCreateMemoryBufferWithContentsOfFile.argtypes = [c_char_p,
            POINTER(c_object_p), POINTER(c_char_p)]
    library.LLVMCreateMemoryBufferWithContentsOfFile.restype = bool

    library.LLVMGetBufferSize.argtypes = [MemoryBuffer]

    library.LLVMDisposeMemoryBuffer.argtypes = [MemoryBuffer]

    # Module declarations
    library.LLVMModuleCreateWithName.argtypes = [c_char_p]
    library.LLVMModuleCreateWithName.restype = c_object_p

    library.LLVMDisposeModule.argtypes = [Module]
    library.LLVMDisposeModule.restype = None

    library.LLVMGetDataLayout.argtypes = [Module]
    library.LLVMGetDataLayout.restype = c_char_p

    library.LLVMSetDataLayout.argtypes = [Module, c_char_p]
    library.LLVMSetDataLayout.restype = None

    library.LLVMGetTarget.argtypes = [Module]
    library.LLVMGetTarget.restype = c_char_p

    library.LLVMSetTarget.argtypes = [Module, c_char_p]
    library.LLVMSetTarget.restype = None

    library.LLVMDumpModule.argtypes = [Module]
    library.LLVMDumpModule.restype = None

    library.LLVMPrintModuleToFile.argtypes = [Module, c_char_p,
                                              POINTER(c_char_p)]
    library.LLVMPrintModuleToFile.restype = bool

def register_enumerations():
    for name, value in enumerations.OpCodes:
        OpCode.register(name, value)

def initialize_llvm():
    c = Context.GetGlobalContext()
    p = PassRegistry()
    lib.LLVMInitializeCore(p)
    lib.LLVMInitializeTransformUtils(p)
    lib.LLVMInitializeScalarOpts(p)
    lib.LLVMInitializeObjCARCOpts(p)
    lib.LLVMInitializeVectorization(p)
    lib.LLVMInitializeInstCombine(p)
    lib.LLVMInitializeIPO(p)
    lib.LLVMInitializeInstrumentation(p)
    lib.LLVMInitializeAnalysis(p)
    lib.LLVMInitializeIPA(p)
    lib.LLVMInitializeCodeGen(p)
    lib.LLVMInitializeTarget(p)

register_library(lib)
register_enumerations()
initialize_llvm()
