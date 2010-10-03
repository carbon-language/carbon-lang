//====- X86Instr3DNow.td - The 3DNow! Instruction Set ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the 3DNow! instruction set, which extends MMX to support
// floating point and also adds a few more random instructions for good measure.
//
//===----------------------------------------------------------------------===//

// FIXME: We don't support any intrinsics for these instructions yet.

class I3DNow<bits<8> o, Format F, dag outs, dag ins, string asm, 
             list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, TB, Requires<[Has3DNow]>,
        Has3DNow0F0FOpcode {
  // FIXME: The disassembler doesn't support 3DNow! yet.
  let isAsmParserOnly = 1;
}


let Constraints = "$src1 = $dst" in {
  // MMXI_binop_rm_int - Simple MMX binary operator based on intrinsic.
  // When this is cleaned up, remove the FIXME from X86RecognizableInstr.cpp.
  multiclass I3DNow_binop_rm<bits<8> opc, string Mnemonic> {
    def rr : I3DNow<opc, MRMSrcReg, (outs VR64:$dst),
                    (ins VR64:$src1, VR64:$src2),
                    !strconcat(Mnemonic, "\t{$src2, $dst|$dst, $src2}"), []>;
    def rm : I3DNow<opc, MRMSrcMem, (outs VR64:$dst),
                    (ins VR64:$src1, i64mem:$src2),
                    !strconcat(Mnemonic, "\t{$src2, $dst|$dst, $src2}"), []>;
  }
}

defm PAVGUSB : I3DNow_binop_rm<0xBF, "pavgusb">;




// TODO: Add support for the rest of the 3DNow! and "3DNowA" instructions.
