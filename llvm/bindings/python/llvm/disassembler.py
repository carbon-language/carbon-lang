#===- disassembler.py - Python LLVM Bindings -----------------*- python -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

from abc import ABCMeta
from abc import abstractmethod

from ctypes import CFUNCTYPE
from ctypes import POINTER
from ctypes import byref
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_ubyte
from ctypes import c_uint64
from ctypes import c_uint
from ctypes import c_void_p
from ctypes import memmove

from .common import CachedProperty
from .common import LLVMObject
from .common import c_object_p
from .common import get_library

__all__ = [
    'DisassemblerByteArraySource',
    'DisassemblerFileSource',
    'DisassemblerSource',
    'Disassembler',
    'Instruction',
    'Operand',
    'Token',
]

callbacks = {}

class DisassemblerSource:
    """Abstract base class for disassembler input.

    This defines the interface to which inputs to the disassembler must
    conform.

    Basically, the disassembler input is a read-only sequence of a finite
    length.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __len__(self):
        """Returns the number of bytes that are available for input."""
        pass

    @abstractmethod
    def get_byte(self, address):
        """Returns the byte at the specified address."""
        pass

    @abstractmethod
    def start_address(self):
        """Returns the address at which to start fetch bytes, as a long."""
        pass

class DisassemblerByteArraySource(DisassemblerSource):
    """A disassembler source for byte arrays."""

    def __init__(self, b):
        self._array = b

    def __len__(self):
        return len(self._array)

    def get_byte(self, address):
        return self._array[address]

    def start_address(self):
        return 0

class DisassemblerFileSource(DisassemblerSource):
    """A disassembler source for file segments.

    This allows you to feed in segments of a file into a Disassembler.
    """

    def __init__(self, filename, start_offset, length=None, end_offset=None,
                 start_address=None):
        """Create a new source from a file.

        A source begins at a specified byte offset and can be defined in terms
        of byte length of the end byte offset.
        """
        if length is None and end_offset is None:
            raise Exception('One of length or end_offset must be defined.')

        self._start_address = start_address
        if self._start_address is None:
            self._start_address = 0

        count = length
        if length is None:
            count = end_offset - start_offset

        with open(filename, 'rb') as fh:
            fh.seek(start_offset)

            # FIXME handle case where read bytes != requested
            self._buf = fh.read(count)

    def __len__(self):
        return len(self._buf)

    def get_byte(self, address):
        return self._buf[address - self._start_address]

    def start_address(self):
        return self._start_address

class Disassembler(LLVMObject):
    """Interface to LLVM's enhanced disassembler.

    The API is slightly different from the C API in that we tightly couple a
    disassembler instance to an input source. This saves an extra level of
    abstraction and makes the Python implementation easier.
    """

    SYNTAX_X86_INTEL = 0
    SYNTAX_X86_ATT = 1
    SYNTAX_ARM_UAL = 2

    def __init__(self, triple, source, syntax=0):
        """Create a new disassembler instance.

        Arguments:

        triple -- str target type (e.g. x86_64-apple-darwin10)
        source -- DisassemblerSource instance to be fed into this disassembler.
        syntax -- The assembly syntax to use. One of the SYNTAX_* class
            constants. e.g. EnhancedDisassembler.SYNTAX_X86_INTEL
        """
        assert isinstance(source, DisassemblerSource)

        ptr = c_object_p()
        result = lib.EDGetDisassembler(byref(ptr), c_char_p(triple),
                                       c_int(syntax))
        if result != 0:
            raise Exception('Non-0 return code.')

        LLVMObject.__init__(self, ptr)

        self._source = source

    def get_instructions(self):
        """Obtain the instructions from the input.

        This is a generator for Instruction instances.

        By default, this will return instructions for the entire source which
        has been defined. It does this by querying the source's start_address()
        method and continues to request instructions until len(source) is
        exhausted.
        """

        # We currently obtain 1 instruction at a time because it is easiest.

        # This serves as our EDByteReaderCallback. It is a proxy between C and
        # the Python DisassemblerSource.
        def byte_reader(dest, address, arg):
            try:
                byte = self._source.get_byte(address)
                memmove(dest, byte, 1)

                return 0
            except:
                return -1

        address = self._source.start_address()
        end_address = address + len(self._source)
        cb = callbacks['byte_reader'](byte_reader)
        while address < end_address:
            ptr = c_object_p()

            result = lib.EDCreateInsts(byref(ptr), c_uint(1), self, cb,
                                       address, c_void_p(None))

            if result != 1:
                raise Exception('Error obtaining instruction at address %d' %
                        address)

            instruction = Instruction(ptr, self)
            yield instruction

            address += instruction.byte_size


class Instruction(LLVMObject):
    """Represents an individual instruction.

    Instruction instances are obtained from Disassembler.get_instructions().
    """
    def __init__(self, ptr, disassembler):
        """Create a new instruction.

        Instructions are created from within this module. You should have no
        need to call this from outside this module.
        """
        assert isinstance(ptr, c_object_p)
        assert isinstance(disassembler, Disassembler)

        LLVMObject.__init__(self, ptr, disposer=lib.EDReleaseInst)
        self._disassembler = disassembler

    def __str__(self):
        s = c_char_p(None)
        result = lib.EDGetInstString(byref(s), self)
        if result != 0:
            raise Exception('Non-0 return code.')

        return s.value

    @CachedProperty
    def byte_size(self):
        result = lib.EDInstByteSize(self)
        if result == -1:
            raise Exception('Error code returned.')

        return result

    @CachedProperty
    def id(self):
        i = c_uint()
        result = lib.EDInstID(byref(i), self)
        if result != 0:
            raise Exception('Non-0 return code.')

        return i.value

    @CachedProperty
    def is_branch(self):
        result = lib.EDInstIsBranch(self)
        if result == -1:
            raise Exception('Error code returned.')

        return result > 0

    @CachedProperty
    def is_move(self):
        result = lib.EDInstIsMove(self)
        if result == -1:
            raise Exception('Error code returned.')

        return result > 0

    @CachedProperty
    def branch_target_id(self):
        result = lib.EDBranchTargetID(self)
        if result == -1:
            raise Exception('Error code returned.')

        return result

    @CachedProperty
    def move_source_id(self):
        result = lib.EDMoveSourceID(self)
        if result == -1:
            raise Exception('Error code returned.')

        return result

    def get_tokens(self):
        """Obtain the tokens in this instruction.

        This is a generator for Token instances.
        """
        count = lib.EDNumTokens(self)
        if count == -1:
            raise Exception('Error code returned.')

        for i in range(0, count):
            ptr = c_object_p()
            result = lib.EDGetToken(byref(ptr), self, c_int(i))
            if result != 0:
                raise Exception('Non-0 return code.')

            yield Token(ptr, self)

    def get_operands(self):
        """Obtain the operands in this instruction.

        This is a generator for Operand instances.
        """
        count = lib.EDNumOperands(self)
        if count == -1:
            raise Exception('Error code returned.')

        for i in range(0, count):
            ptr = c_object_p()
            result = lib.EDGetOperand(byref(ptr), self, c_int(i))
            if result != 0:
                raise Exception('Non-0 return code.')

            yield Operand(ptr, self)

class Token(LLVMObject):
    def __init__(self, ptr, instruction):
        assert isinstance(ptr, c_object_p)
        assert isinstance(instruction, Instruction)

        LLVMObject.__init__(self, ptr)

        self._instruction = instruction

    def __str__(self):
        s = c_char_p(None)
        result = lib.EDGetTokenString(byref(s), self)
        if result != 0:
            raise Exception('Non-0 return code.')

        return s.value

    @CachedProperty
    def operand_index(self):
        result = lib.EDOperandIndexForToken(self)
        if result == -1:
            raise Exception('Error code returned.')

        return result

    @CachedProperty
    def is_whitespace(self):
        result = lib.EDTokenIsWhitespace(self)
        if result == -1:
            raise Exception('Error code returned.')

        return result > 0

    @CachedProperty
    def is_punctuation(self):
        result = lib.EDTokenIsPunctuation(self)
        if result == -1:
            raise Exception('Error code returned.')

        return result > 0

    @CachedProperty
    def is_opcode(self):
        result = lib.EDTokenIsOpcode(self)
        if result == -1:
            raise Exception('Error code returned.')

        return result > 0

    @CachedProperty
    def is_literal(self):
        result = lib.EDTokenIsLiteral(self)
        if result == -1:
            raise Exception('Error code returned.')

        return result > 0

    @CachedProperty
    def is_register(self):
        result = lib.EDTokenIsRegister(self)
        if result == -1:
            raise Exception('Error code returned.')

        return result > 0

    @CachedProperty
    def is_negative_literal(self):
        result = lib.EDTokenIsNegativeLiteral(self)
        if result == -1:
            raise Exception('Error code returned.')

        return result > 0

    @CachedProperty
    def absolute_value(self):
        value = c_uint64()
        result = lib.EDLiteralTokenAbsoluteValue(byref(value), self)
        if result != 0:
            raise Exception('Non-0 return code.')

        return value

    @CachedProperty
    def register_value(self):
        value = c_uint()
        result = lib.EDRegisterTokenValue(byref(value), self)
        if result != 0:
            raise Exception('Non-0 return code.')

        return value

class Operand(LLVMObject):
    """Represents an operand in an instruction.

    FIXME support register evaluation.
    """
    def __init__(self, ptr, instruction):
        assert isinstance(ptr, c_object_p)
        assert isinstance(instruction, Instruction)

        LLVMObject.__init__(self, ptr)

        self._instruction = instruction

    @CachedProperty
    def is_register(self):
        result = lib.EDOperandIsRegister(self)
        if result == -1:
            raise Exception('Error code returned.')

        return result > 0

    @CachedProperty
    def is_immediate(self):
        result = lib.EDOperandIsImmediate(self)
        if result == -1:
            raise Exception('Error code returned.')

        return result > 0

    @CachedProperty
    def is_memory(self):
        result = lib.EDOperandIsMemory(self)
        if result == -1:
            raise Exception('Error code returned.')

        return result > 0

    @CachedProperty
    def register_value(self):
        value = c_uint()
        result = lib.EDRegisterOperandValue(byref(value), self)
        if result != 0:
            raise Exception('Non-0 return code.')

        return value

    @CachedProperty
    def immediate_value(self):
        value = c_uint64()
        result = lib.EDImmediateOperandValue(byref(value), self)
        if result != 0:
            raise Exception('Non-0 return code.')

        return value

def register_library(library):
    library.EDGetDisassembler.argtypes = [POINTER(c_object_p), c_char_p, c_int]
    library.EDGetDisassembler.restype = c_int

    library.EDGetRegisterName.argtypes = [POINTER(c_char_p), Disassembler,
            c_uint]
    library.EDGetRegisterName.restype = c_int

    library.EDRegisterIsStackPointer.argtypes = [Disassembler, c_uint]
    library.EDRegisterIsStackPointer.restype = c_int

    library.EDRegisterIsProgramCounter.argtypes = [Disassembler, c_uint]
    library.EDRegisterIsProgramCounter.restype = c_int

    library.EDCreateInsts.argtypes = [POINTER(c_object_p), c_uint,
            Disassembler, callbacks['byte_reader'], c_uint64, c_void_p]
    library.EDCreateInsts.restype = c_uint

    library.EDReleaseInst.argtypes = [Instruction]

    library.EDInstByteSize.argtypes = [Instruction]
    library.EDInstByteSize.restype = c_int

    library.EDGetInstString.argtypes = [POINTER(c_char_p), Instruction]
    library.EDGetInstString.restype = c_int

    library.EDInstID.argtypes = [POINTER(c_uint), Instruction]
    library.EDInstID.restype = c_int

    library.EDInstIsBranch.argtypes = [Instruction]
    library.EDInstIsBranch.restype = c_int

    library.EDInstIsMove.argtypes = [Instruction]
    library.EDInstIsMove.restype = c_int

    library.EDBranchTargetID.argtypes = [Instruction]
    library.EDBranchTargetID.restype = c_int

    library.EDMoveSourceID.argtypes = [Instruction]
    library.EDMoveSourceID.restype = c_int

    library.EDMoveTargetID.argtypes = [Instruction]
    library.EDMoveTargetID.restype = c_int

    library.EDNumTokens.argtypes = [Instruction]
    library.EDNumTokens.restype = c_int

    library.EDGetToken.argtypes = [POINTER(c_object_p), Instruction, c_int]
    library.EDGetToken.restype = c_int

    library.EDGetTokenString.argtypes = [POINTER(c_char_p), Token]
    library.EDGetTokenString.restype = c_int

    library.EDOperandIndexForToken.argtypes = [Token]
    library.EDOperandIndexForToken.restype = c_int

    library.EDTokenIsWhitespace.argtypes = [Token]
    library.EDTokenIsWhitespace.restype = c_int

    library.EDTokenIsPunctuation.argtypes = [Token]
    library.EDTokenIsPunctuation.restype = c_int

    library.EDTokenIsOpcode.argtypes = [Token]
    library.EDTokenIsOpcode.restype = c_int

    library.EDTokenIsLiteral.argtypes = [Token]
    library.EDTokenIsLiteral.restype = c_int

    library.EDTokenIsRegister.argtypes = [Token]
    library.EDTokenIsRegister.restype = c_int

    library.EDTokenIsNegativeLiteral.argtypes = [Token]
    library.EDTokenIsNegativeLiteral.restype = c_int

    library.EDLiteralTokenAbsoluteValue.argtypes = [POINTER(c_uint64), Token]
    library.EDLiteralTokenAbsoluteValue.restype = c_int

    library.EDRegisterTokenValue.argtypes = [POINTER(c_uint), Token]
    library.EDRegisterTokenValue.restype = c_int

    library.EDNumOperands.argtypes = [Instruction]
    library.EDNumOperands.restype = c_int

    library.EDGetOperand.argtypes = [POINTER(c_object_p), Instruction, c_int]
    library.EDGetOperand.restype = c_int

    library.EDOperandIsRegister.argtypes = [Operand]
    library.EDOperandIsRegister.restype = c_int

    library.EDOperandIsImmediate.argtypes = [Operand]
    library.EDOperandIsImmediate.restype = c_int

    library.EDOperandIsMemory.argtypes = [Operand]
    library.EDOperandIsMemory.restype = c_int

    library.EDRegisterOperandValue.argtypes = [POINTER(c_uint), Operand]
    library.EDRegisterOperandValue.restype = c_int

    library.EDImmediateOperandValue.argtypes = [POINTER(c_uint64), Operand]
    library.EDImmediateOperandValue.restype = c_int

    library.EDEvaluateOperand.argtypes = [c_uint64, Operand,
        callbacks['register_reader'], c_void_p]
    library.EDEvaluateOperand.restype = c_int

# Enhanced disassembler.
callbacks['byte_reader'] = CFUNCTYPE(c_int, POINTER(c_ubyte), c_uint64,
                                     c_void_p)
callbacks['register_reader'] = CFUNCTYPE(c_int, POINTER(c_uint64), c_uint,
                                         c_void_p)

lib = get_library()
register_library(lib)
