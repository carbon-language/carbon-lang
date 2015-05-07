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
class ADD_MMR6_ENC : ARITH_FM_MMR6<"add", 0x110>;
class ADDIU_MMR6_ENC : ADDI_FM_MMR6<"addiu", 0xc>;
class ADDU_MMR6_ENC : ARITH_FM_MMR6<"addu", 0x150>;
class BALC_MMR6_ENC  : BRANCH_OFF26_FM<0b101101>;
class BC_MMR6_ENC : BRANCH_OFF26_FM<0b100101>;
class BITSWAP_MMR6_ENC : POOL32A_BITSWAP_FM_MMR6<0b101100>;
class CACHE_MMR6_ENC : CACHE_PREF_FM_MMR6<0b001000, 0b0110>;
class JIALC_MMR6_ENC : JMP_IDX_COMPACT_FM<0b100000>;
class JIC_MMR6_ENC   : JMP_IDX_COMPACT_FM<0b101000>;
class MUL_MMR6_ENC : ARITH_FM_MMR6<"mul", 0x18>;
class MUH_MMR6_ENC : ARITH_FM_MMR6<"muh", 0x58>;
class MULU_MMR6_ENC : ARITH_FM_MMR6<"mulu", 0x98>;
class MUHU_MMR6_ENC : ARITH_FM_MMR6<"muhu", 0xd8>;
class PREF_MMR6_ENC : CACHE_PREF_FM_MMR6<0b011000, 0b0010>;
class SUB_MMR6_ENC : ARITH_FM_MMR6<"sub", 0x190>;
class SUBU_MMR6_ENC : ARITH_FM_MMR6<"subu", 0x1d0>;

//===----------------------------------------------------------------------===//
//
// Instruction Descriptions
//
//===----------------------------------------------------------------------===//

class ADD_MMR6_DESC : ArithLogicR<"add", GPR32Opnd>;
class ADDIU_MMR6_DESC : ArithLogicI<"addiu", simm16, GPR32Opnd>;
class ADDU_MMR6_DESC : ArithLogicR<"addu", GPR32Opnd>;
class MUL_MMR6_DESC : ArithLogicR<"mul", GPR32Opnd>;
class MUH_MMR6_DESC : ArithLogicR<"muh", GPR32Opnd>;
class MULU_MMR6_DESC : ArithLogicR<"mulu", GPR32Opnd>;
class MUHU_MMR6_DESC : ArithLogicR<"muhu", GPR32Opnd>;

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
class SUB_MMR6_DESC : ArithLogicR<"sub", GPR32Opnd>;
class SUBU_MMR6_DESC : ArithLogicR<"subu", GPR32Opnd>;

class BITSWAP_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rt");
  list<dag> Pattern = [];
}

class BITSWAP_MMR6_DESC : BITSWAP_MMR6_DESC_BASE<"bitswap", GPR32Opnd>;

class CACHE_HINT_MMR6_DESC<string instr_asm, Operand MemOpnd,
                           RegisterOperand GPROpnd> : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins MemOpnd:$addr, uimm5:$hint);
  string AsmString = !strconcat(instr_asm, "\t$hint, $addr");
  list<dag> Pattern = [];
  string DecoderMethod = "DecodeCacheOpMM";
}

class CACHE_MMR6_DESC : CACHE_HINT_MMR6_DESC<"cache", mem_mm_12, GPR32Opnd>;
class PREF_MMR6_DESC : CACHE_HINT_MMR6_DESC<"pref", mem_mm_12, GPR32Opnd>;

class JMP_MMR6_IDX_COMPACT_DESC_BASE<string opstr, DAGOperand opnd,
                                     RegisterOperand GPROpnd>
    : MMR6Arch<opstr> {
  dag InOperandList = (ins GPROpnd:$rt, opnd:$offset);
  string AsmString = !strconcat(opstr, "\t$rt, $offset");
  list<dag> Pattern = [];
  bit isTerminator = 1;
  bit hasDelaySlot = 0;
}

class JIALC_MMR6_DESC : JMP_MMR6_IDX_COMPACT_DESC_BASE<"jialc", calloffset16,
                                                       GPR32Opnd> {
  bit isCall = 1;
  list<Register> Defs = [RA];
}

class JIC_MMR6_DESC : JMP_MMR6_IDX_COMPACT_DESC_BASE<"jic", jmpoffset16,
                                                     GPR32Opnd> {
  bit isBarrier = 1;
  list<Register> Defs = [AT];
}

//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

let DecoderNamespace = "MicroMips32r6" in {
def ADD_MMR6 : StdMMR6Rel, ADD_MMR6_DESC, ADD_MMR6_ENC, ISA_MICROMIPS32R6;
def ADDIU_MMR6 : StdMMR6Rel, ADDIU_MMR6_DESC, ADDIU_MMR6_ENC, ISA_MICROMIPS32R6;
def ADDU_MMR6 : StdMMR6Rel, ADDU_MMR6_DESC, ADDU_MMR6_ENC, ISA_MICROMIPS32R6;
def BALC_MMR6 : R6MMR6Rel, BALC_MMR6_ENC, BALC_MMR6_DESC, ISA_MICROMIPS32R6;
def BC_MMR6 : R6MMR6Rel, BC_MMR6_ENC, BC_MMR6_DESC, ISA_MICROMIPS32R6;
def BITSWAP_MMR6 : R6MMR6Rel, BITSWAP_MMR6_ENC, BITSWAP_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def CACHE_MMR6 : R6MMR6Rel, CACHE_MMR6_ENC, CACHE_MMR6_DESC, ISA_MICROMIPS32R6;
def JIALC_MMR6 : R6MMR6Rel, JIALC_MMR6_ENC, JIALC_MMR6_DESC, ISA_MICROMIPS32R6;
def JIC_MMR6 : R6MMR6Rel, JIC_MMR6_ENC, JIC_MMR6_DESC, ISA_MICROMIPS32R6;
def MUL_MMR6 : R6MMR6Rel, MUL_MMR6_DESC, MUL_MMR6_ENC, ISA_MICROMIPS32R6;
def MUH_MMR6 : R6MMR6Rel, MUH_MMR6_DESC, MUH_MMR6_ENC, ISA_MICROMIPS32R6;
def MULU_MMR6 : R6MMR6Rel, MULU_MMR6_DESC, MULU_MMR6_ENC, ISA_MICROMIPS32R6;
def MUHU_MMR6 : R6MMR6Rel, MUHU_MMR6_DESC, MUHU_MMR6_ENC, ISA_MICROMIPS32R6;
def PREF_MMR6 : R6MMR6Rel, PREF_MMR6_ENC, PREF_MMR6_DESC, ISA_MICROMIPS32R6;
def SUB_MMR6 : StdMMR6Rel, SUB_MMR6_DESC, SUB_MMR6_ENC, ISA_MICROMIPS32R6;
def SUBU_MMR6 : StdMMR6Rel, SUBU_MMR6_DESC, SUBU_MMR6_ENC, ISA_MICROMIPS32R6;
}
