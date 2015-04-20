//=- MicroMips32r6InstrInfo.td - MicroMips r6 Instruction Information -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes microMIPSr6 instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Instruction Encodings
//
//===----------------------------------------------------------------------===//

class BALC_MMR6_ENC  : BRANCH_OFF26_FM<0b101101>;
class BC_MMR6_ENC : BRANCH_OFF26_FM<0b100101>;

//===----------------------------------------------------------------------===//
//
// Instruction Descriptions
//
//===----------------------------------------------------------------------===//

class BC_MMR6_DESC_BASE<string instr_asm, DAGOperand opnd>
    : BRANCH_DESC_BASE, MMR6Arch<instr_asm> {
  dag InOperandList = (ins opnd:$offset);
  dag OutOperandList = (outs);
  string AsmString = !strconcat(instr_asm, "\t$offset");
  bit isBarrier = 1;
}

class BALC_MMR6_DESC : BC_MMR6_DESC_BASE<"balc", brtarget26> {
  bit isCall = 1;
  list<Register> Defs = [RA];
}
class BC_MMR6_DESC : BC_MMR6_DESC_BASE<"bc", brtarget26>;

//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

def BALC_MMR6 : R6MMR6Rel, BALC_MMR6_ENC, BALC_MMR6_DESC, ISA_MICROMIPS32R6;
def BC_MMR6 : R6MMR6Rel, BC_MMR6_ENC, BC_MMR6_DESC, ISA_MICROMIPS32R6;
