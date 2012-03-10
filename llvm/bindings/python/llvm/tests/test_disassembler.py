from unittest import expectedFailure
from unittest import skip

from .base import TestBase
from ..disassembler import DisassemblerByteArraySource
from ..disassembler import DisassemblerFileSource
from ..disassembler import Disassembler
from ..object import ObjectFile

class TestDisassembler(TestBase):
    def test_simple(self):
        sequence = '\x67\xe3\x81' # jcxz -127
        triple = 'i686-apple-darwin9'

        source = DisassemblerByteArraySource(sequence)

        disassembler = Disassembler(triple, source)
        instructions = list(disassembler.get_instructions())

        self.assertEqual(len(instructions), 1)

        i = instructions[0]
        self.assertEqual(str(i), '\tjcxz\t-127\n')
        self.assertEqual(i.byte_size, 3)
        self.assertEqual(i.id, 1032)
        self.assertTrue(i.is_branch)
        self.assertFalse(i.is_move)
        self.assertEqual(i.branch_target_id, 0)

        tokens = list(i.get_tokens())
        self.assertEqual(len(tokens), 4)
        token = tokens[0]
        self.assertEqual(str(token), 'jcxz')
        self.assertFalse(token.is_whitespace)
        self.assertFalse(token.is_punctuation)
        self.assertTrue(token.is_opcode)
        self.assertFalse(token.is_literal)
        self.assertFalse(token.is_register)

        self.assertTrue(tokens[1].is_whitespace)

        operands = list(i.get_operands())
        self.assertEqual(len(operands), 1)

        # TODO implement operand tests

    @skip('This test is horribly broken and probably not even correct.')
    def test_read_instructions(self):
        filename = self.get_test_binary()
        o = ObjectFile(filename=filename)

        for symbol in o.get_symbols():
            address = symbol.address
            offset = symbol.file_offset
            size = symbol.size

            source = DisassemblerFileSource(filename, offset, length=size,
                                            start_address=address)

            disassembler = Disassembler('x86-generic-gnu-linux', source)
            for instruction in disassembler.get_instructions():
                print instruction
