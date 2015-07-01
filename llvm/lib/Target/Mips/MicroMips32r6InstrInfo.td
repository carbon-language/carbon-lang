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
class ADDIUPC_MMR6_ENC : PCREL19_FM_MMR6<0b00>;
class ALUIPC_MMR6_ENC : PCREL16_FM_MMR6<0b11111>;
class AND_MMR6_ENC : ARITH_FM_MMR6<"and", 0x250>;
class ANDI_MMR6_ENC : ADDI_FM_MMR6<"andi", 0x34>;
class AUIPC_MMR6_ENC  : PCREL16_FM_MMR6<0b11110>;
class ALIGN_MMR6_ENC : POOL32A_ALIGN_FM_MMR6<0b011111>;
class AUI_MMR6_ENC : AUI_FM_MMR6;
class BALC_MMR6_ENC  : BRANCH_OFF26_FM<0b101101>;
class BC_MMR6_ENC : BRANCH_OFF26_FM<0b100101>;
class BITSWAP_MMR6_ENC : POOL32A_BITSWAP_FM_MMR6<0b101100>;
class BRK_MMR6_ENC : BREAK_MMR6_ENC<"break">;
class BEQZALC_MMR6_ENC : CMP_BRANCH_1R_RT_OFF16_FM_MMR6<0b011101>;
class BNEZALC_MMR6_ENC : CMP_BRANCH_1R_RT_OFF16_FM_MMR6<0b011111>;
class BGTZALC_MMR6_ENC : CMP_BRANCH_1R_RT_OFF16_FM_MMR6<0b111000>;
class BLTZALC_MMR6_ENC : CMP_BRANCH_1R_BOTH_OFF16_FM_MMR6<0b111000>;
class BGEZALC_MMR6_ENC : CMP_BRANCH_1R_BOTH_OFF16_FM_MMR6<0b110000>;
class BLEZALC_MMR6_ENC : CMP_BRANCH_1R_RT_OFF16_FM_MMR6<0b110000>;
class CACHE_MMR6_ENC : CACHE_PREF_FM_MMR6<0b001000, 0b0110>;
class CLO_MMR6_ENC : POOL32A_2R_FM_MMR6<0b0100101100>;
class CLZ_MMR6_ENC : SPECIAL_2R_FM_MMR6<0b010000>;
class DIV_MMR6_ENC : ARITH_FM_MMR6<"div", 0x118>;
class DIVU_MMR6_ENC : ARITH_FM_MMR6<"divu", 0x198>;
class EHB_MMR6_ENC : BARRIER_MMR6_ENC<"ehb", 0x3>;
class EI_MMR6_ENC : EIDI_MMR6_ENC<"ei", 0x15d>;
class ERET_MMR6_ENC : ERET_FM_MMR6<"eret">;
class ERETNC_MMR6_ENC : ERETNC_FM_MMR6<"eretnc">;
class JIALC_MMR6_ENC : JMP_IDX_COMPACT_FM<0b100000>;
class JIC_MMR6_ENC   : JMP_IDX_COMPACT_FM<0b101000>;
class LSA_MMR6_ENC : POOL32A_LSA_FM<0b001111>;
class LWPC_MMR6_ENC  : PCREL19_FM_MMR6<0b01>;
class MOD_MMR6_ENC : ARITH_FM_MMR6<"mod", 0x158>;
class MODU_MMR6_ENC : ARITH_FM_MMR6<"modu", 0x1d8>;
class MUL_MMR6_ENC : ARITH_FM_MMR6<"mul", 0x18>;
class MUH_MMR6_ENC : ARITH_FM_MMR6<"muh", 0x58>;
class MULU_MMR6_ENC : ARITH_FM_MMR6<"mulu", 0x98>;
class MUHU_MMR6_ENC : ARITH_FM_MMR6<"muhu", 0xd8>;
class NOR_MMR6_ENC : ARITH_FM_MMR6<"nor", 0x2d0>;
class OR_MMR6_ENC : ARITH_FM_MMR6<"or", 0x290>;
class ORI_MMR6_ENC : ADDI_FM_MMR6<"ori", 0x14>;
class PREF_MMR6_ENC : CACHE_PREF_FM_MMR6<0b011000, 0b0010>;
class SEB_MMR6_ENC : SIGN_EXTEND_FM_MMR6<"seb", 0b0010101100>;
class SEH_MMR6_ENC : SIGN_EXTEND_FM_MMR6<"seh", 0b0011101100>;
class SELEQZ_MMR6_ENC : POOL32A_FM_MMR6<0b0101000000>;
class SELNEZ_MMR6_ENC : POOL32A_FM_MMR6<0b0110000000>;
class SLL_MMR6_ENC : SHIFT_MMR6_ENC<"sll", 0x00, 0b0>;
class SUB_MMR6_ENC : ARITH_FM_MMR6<"sub", 0x190>;
class SUBU_MMR6_ENC : ARITH_FM_MMR6<"subu", 0x1d0>;
class XOR_MMR6_ENC : ARITH_FM_MMR6<"xor", 0x310>;
class XORI_MMR6_ENC : ADDI_FM_MMR6<"xori", 0x1c>;

class CMP_CBR_RT_Z_MMR6_DESC_BASE<string instr_asm, DAGOperand opnd,
                                  RegisterOperand GPROpnd>
    : BRANCH_DESC_BASE, MMR6Arch<instr_asm> {
  dag InOperandList = (ins GPROpnd:$rt, opnd:$offset);
  dag OutOperandList = (outs);
  string AsmString = !strconcat(instr_asm, "\t$rt, $offset");
  list<Register> Defs = [AT];
}

class BEQZALC_MMR6_DESC : CMP_CBR_RT_Z_MMR6_DESC_BASE<"beqzalc", brtarget_mm,
                                                      GPR32Opnd> {
  list<Register> Defs = [RA];
}

class BGEZALC_MMR6_DESC : CMP_CBR_RT_Z_MMR6_DESC_BASE<"bgezalc", brtarget_mm,
                                                      GPR32Opnd> {
  list<Register> Defs = [RA];
}

class BGTZALC_MMR6_DESC : CMP_CBR_RT_Z_MMR6_DESC_BASE<"bgtzalc", brtarget_mm,
                                                      GPR32Opnd> {
  list<Register> Defs = [RA];
}

class BLEZALC_MMR6_DESC : CMP_CBR_RT_Z_MMR6_DESC_BASE<"blezalc", brtarget_mm,
                                                      GPR32Opnd> {
  list<Register> Defs = [RA];
}

class BLTZALC_MMR6_DESC : CMP_CBR_RT_Z_MMR6_DESC_BASE<"bltzalc", brtarget_mm,
                                                      GPR32Opnd> {
  list<Register> Defs = [RA];
}

class BNEZALC_MMR6_DESC : CMP_CBR_RT_Z_MMR6_DESC_BASE<"bnezalc", brtarget_mm,
                                                      GPR32Opnd> {
  list<Register> Defs = [RA];
}

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

class BRK_MMR6_DESC : BRK_FT<"break">;

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

class CLO_CLZ_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rt);
  dag InOperandList = (ins GPROpnd:$rs);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs");
}

class CLO_MMR6_DESC : CLO_CLZ_MMR6_DESC_BASE<"clo", GPR32Opnd>;
class CLZ_MMR6_DESC : CLO_CLZ_MMR6_DESC_BASE<"clz", GPR32Opnd>;

class EHB_MMR6_DESC : Barrier<"ehb">;
class EI_MMR6_DESC : DEI_FT<"ei", GPR32Opnd>;

class ERET_MMR6_DESC : ER_FT<"eret">;
class ERETNC_MMR6_DESC : ER_FT<"eretnc">;

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

class ALIGN_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                      Operand ImmOpnd>  : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt, ImmOpnd:$bp);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt, $bp");
  list<dag> Pattern = [];
}

class ALIGN_MMR6_DESC : ALIGN_MMR6_DESC_BASE<"align", GPR32Opnd, uimm2>;

class AUI_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rt);
  dag InOperandList = (ins GPROpnd:$rs, simm16:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs, $imm");
  list<dag> Pattern = [];
}

class AUI_MMR6_DESC : AUI_MMR6_DESC_BASE<"aui", GPR32Opnd>;

class SEB_MMR6_DESC : SignExtInReg<"seb", i8, GPR32Opnd, II_SEB>;
class SEH_MMR6_DESC : SignExtInReg<"seh", i16, GPR32Opnd, II_SEH>;
class ALUIPC_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rt);
  dag InOperandList = (ins simm16:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rt, $imm");
  list<dag> Pattern = [];
}

class ALUIPC_MMR6_DESC : ALUIPC_MMR6_DESC_BASE<"aluipc", GPR32Opnd>;
class AUIPC_MMR6_DESC : ALUIPC_MMR6_DESC_BASE<"auipc", GPR32Opnd>;

class LSA_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                         Operand ImmOpnd> : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt, ImmOpnd:$imm2);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs, $rd, $imm2");
  list<dag> Pattern = [];
}

class LSA_MMR6_DESC : LSA_MMR6_DESC_BASE<"lsa", GPR32Opnd, uimm2>;

class PCREL_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                           Operand ImmOpnd> : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rt);
  dag InOperandList = (ins ImmOpnd:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rt, $imm");
  list<dag> Pattern = [];
}

class ADDIUPC_MMR6_DESC : PCREL_MMR6_DESC_BASE<"addiupc", GPR32Opnd, simm19_lsl2>;
class LWPC_MMR6_DESC: PCREL_MMR6_DESC_BASE<"lwpc", GPR32Opnd, simm19_lsl2>;

class SELEQNE_Z_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt");
  list<dag> Pattern = [];
}

class SELEQZ_MMR6_DESC : SELEQNE_Z_MMR6_DESC_BASE<"seleqz", GPR32Opnd>;
class SELNEZ_MMR6_DESC : SELEQNE_Z_MMR6_DESC_BASE<"selnez", GPR32Opnd>;
class SLL_MMR6_DESC : shift_rotate_imm<"sll", uimm5, GPR32Opnd, II_SLL>;
class DIV_MMR6_DESC : ArithLogicR<"div", GPR32Opnd>;
class DIVU_MMR6_DESC : ArithLogicR<"divu", GPR32Opnd>;
class MOD_MMR6_DESC : ArithLogicR<"mod", GPR32Opnd>;
class MODU_MMR6_DESC : ArithLogicR<"modu", GPR32Opnd>;
class AND_MMR6_DESC : ArithLogicR<"and", GPR32Opnd>;
class ANDI_MMR6_DESC : ArithLogicI<"andi", simm16, GPR32Opnd>;
class NOR_MMR6_DESC : ArithLogicR<"nor", GPR32Opnd>;
class OR_MMR6_DESC : ArithLogicR<"or", GPR32Opnd>;
class ORI_MMR6_DESC : ArithLogicI<"ori", simm16, GPR32Opnd>;
class XOR_MMR6_DESC : ArithLogicR<"xor", GPR32Opnd>;
class XORI_MMR6_DESC : ArithLogicI<"xori", simm16, GPR32Opnd>;

//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

let DecoderNamespace = "MicroMips32r6" in {
def ADD_MMR6 : StdMMR6Rel, ADD_MMR6_DESC, ADD_MMR6_ENC, ISA_MICROMIPS32R6;
def ADDIU_MMR6 : StdMMR6Rel, ADDIU_MMR6_DESC, ADDIU_MMR6_ENC, ISA_MICROMIPS32R6;
def ADDU_MMR6 : StdMMR6Rel, ADDU_MMR6_DESC, ADDU_MMR6_ENC, ISA_MICROMIPS32R6;
def ADDIUPC_MMR6 : R6MMR6Rel, ADDIUPC_MMR6_ENC, ADDIUPC_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def ALUIPC_MMR6 : R6MMR6Rel, ALUIPC_MMR6_ENC, ALUIPC_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def AND_MMR6 : StdMMR6Rel, AND_MMR6_DESC, AND_MMR6_ENC, ISA_MICROMIPS32R6;
def ANDI_MMR6 : StdMMR6Rel, ANDI_MMR6_DESC, ANDI_MMR6_ENC, ISA_MICROMIPS32R6;
def AUIPC_MMR6 : R6MMR6Rel, AUIPC_MMR6_ENC, AUIPC_MMR6_DESC, ISA_MICROMIPS32R6;
def ALIGN_MMR6 : R6MMR6Rel, ALIGN_MMR6_ENC, ALIGN_MMR6_DESC, ISA_MICROMIPS32R6;
def AUI_MMR6 : R6MMR6Rel, AUI_MMR6_ENC, AUI_MMR6_DESC, ISA_MICROMIPS32R6;
def BALC_MMR6 : R6MMR6Rel, BALC_MMR6_ENC, BALC_MMR6_DESC, ISA_MICROMIPS32R6;
def BC_MMR6 : R6MMR6Rel, BC_MMR6_ENC, BC_MMR6_DESC, ISA_MICROMIPS32R6;
def BITSWAP_MMR6 : R6MMR6Rel, BITSWAP_MMR6_ENC, BITSWAP_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def BEQZALC_MMR6 : R6MMR6Rel, BEQZALC_MMR6_ENC, BEQZALC_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def BGEZALC_MMR6 : R6MMR6Rel, BGEZALC_MMR6_ENC, BGEZALC_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def BGTZALC_MMR6 : R6MMR6Rel, BGTZALC_MMR6_ENC, BGTZALC_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def BLEZALC_MMR6 : R6MMR6Rel, BLEZALC_MMR6_ENC, BLEZALC_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def BLTZALC_MMR6 : R6MMR6Rel, BLTZALC_MMR6_ENC, BLTZALC_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def BNEZALC_MMR6 : R6MMR6Rel, BNEZALC_MMR6_ENC, BNEZALC_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def BREAK_MMR6 : StdMMR6Rel, BRK_MMR6_DESC, BRK_MMR6_ENC, ISA_MICROMIPS32R6;
def CACHE_MMR6 : R6MMR6Rel, CACHE_MMR6_ENC, CACHE_MMR6_DESC, ISA_MICROMIPS32R6;
def CLO_MMR6 : R6MMR6Rel, CLO_MMR6_ENC, CLO_MMR6_DESC, ISA_MICROMIPS32R6;
def CLZ_MMR6 : R6MMR6Rel, CLZ_MMR6_ENC, CLZ_MMR6_DESC, ISA_MICROMIPS32R6;
def DIV_MMR6 : R6MMR6Rel, DIV_MMR6_DESC, DIV_MMR6_ENC, ISA_MICROMIPS32R6;
def DIVU_MMR6 : R6MMR6Rel, DIVU_MMR6_DESC, DIVU_MMR6_ENC, ISA_MICROMIPS32R6;
def EHB_MMR6 : StdMMR6Rel, EHB_MMR6_DESC, EHB_MMR6_ENC, ISA_MICROMIPS32R6;
def EI_MMR6 : StdMMR6Rel, EI_MMR6_DESC, EI_MMR6_ENC, ISA_MICROMIPS32R6;
def ERET_MMR6 : R6MMR6Rel, ERET_MMR6_DESC, ERET_MMR6_ENC, ISA_MICROMIPS32R6;
def ERETNC_MMR6 : R6MMR6Rel, ERETNC_MMR6_DESC, ERETNC_MMR6_ENC,
                  ISA_MICROMIPS32R6;
def JIALC_MMR6 : R6MMR6Rel, JIALC_MMR6_ENC, JIALC_MMR6_DESC, ISA_MICROMIPS32R6;
def JIC_MMR6 : R6MMR6Rel, JIC_MMR6_ENC, JIC_MMR6_DESC, ISA_MICROMIPS32R6;
def LSA_MMR6 : R6MMR6Rel, LSA_MMR6_ENC, LSA_MMR6_DESC, ISA_MICROMIPS32R6;
def LWPC_MMR6 : R6MMR6Rel, LWPC_MMR6_ENC, LWPC_MMR6_DESC, ISA_MICROMIPS32R6;
def MOD_MMR6 : R6MMR6Rel, MOD_MMR6_DESC, MOD_MMR6_ENC, ISA_MICROMIPS32R6;
def MODU_MMR6 : R6MMR6Rel, MODU_MMR6_DESC, MODU_MMR6_ENC, ISA_MICROMIPS32R6;
def MUL_MMR6 : R6MMR6Rel, MUL_MMR6_DESC, MUL_MMR6_ENC, ISA_MICROMIPS32R6;
def MUH_MMR6 : R6MMR6Rel, MUH_MMR6_DESC, MUH_MMR6_ENC, ISA_MICROMIPS32R6;
def MULU_MMR6 : R6MMR6Rel, MULU_MMR6_DESC, MULU_MMR6_ENC, ISA_MICROMIPS32R6;
def MUHU_MMR6 : R6MMR6Rel, MUHU_MMR6_DESC, MUHU_MMR6_ENC, ISA_MICROMIPS32R6;
def NOR_MMR6 : StdMMR6Rel, NOR_MMR6_DESC, NOR_MMR6_ENC, ISA_MICROMIPS32R6;
def OR_MMR6 : StdMMR6Rel, OR_MMR6_DESC, OR_MMR6_ENC, ISA_MICROMIPS32R6;
def ORI_MMR6 : StdMMR6Rel, ORI_MMR6_DESC, ORI_MMR6_ENC, ISA_MICROMIPS32R6;
def PREF_MMR6 : R6MMR6Rel, PREF_MMR6_ENC, PREF_MMR6_DESC, ISA_MICROMIPS32R6;
def SEB_MMR6 : StdMMR6Rel, SEB_MMR6_DESC, SEB_MMR6_ENC, ISA_MICROMIPS32R6;
def SEH_MMR6 : StdMMR6Rel, SEH_MMR6_DESC, SEH_MMR6_ENC, ISA_MICROMIPS32R6;
def SELEQZ_MMR6 : R6MMR6Rel, SELEQZ_MMR6_ENC, SELEQZ_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def SELNEZ_MMR6 : R6MMR6Rel, SELNEZ_MMR6_ENC, SELNEZ_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def SLL_MMR6 : StdMMR6Rel, SLL_MMR6_DESC, SLL_MMR6_ENC, ISA_MICROMIPS32R6;
def SUB_MMR6 : StdMMR6Rel, SUB_MMR6_DESC, SUB_MMR6_ENC, ISA_MICROMIPS32R6;
def SUBU_MMR6 : StdMMR6Rel, SUBU_MMR6_DESC, SUBU_MMR6_ENC, ISA_MICROMIPS32R6;
def XOR_MMR6 : StdMMR6Rel, XOR_MMR6_DESC, XOR_MMR6_ENC, ISA_MICROMIPS32R6;
def XORI_MMR6 : StdMMR6Rel, XORI_MMR6_DESC, XORI_MMR6_ENC, ISA_MICROMIPS32R6;
}

//===----------------------------------------------------------------------===//
//
// MicroMips instruction aliases
//
//===----------------------------------------------------------------------===//

def : MipsInstAlias<"ei", (EI_MMR6 ZERO), 1>, ISA_MICROMIPS32R6;
def : MipsInstAlias<"nop", (SLL_MMR6 ZERO, ZERO, 0), 1>, ISA_MICROMIPS32R6;
