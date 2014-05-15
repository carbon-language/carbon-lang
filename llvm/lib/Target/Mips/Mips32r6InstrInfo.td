//=- Mips32r6InstrInfo.td - Mips32r6 Instruction Information -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips32r6 instructions.
//
//===----------------------------------------------------------------------===//

include "Mips32r6InstrFormats.td"

// Notes about removals/changes from MIPS32r6:
// Unclear: ssnop
// Reencoded: cache, pref
// Reencoded: clo, clz
// Reencoded: jr -> jalr
// Reencoded: jr.hb -> jalr.hb
// Reencoded: ldc2
// Reencoded: ll, sc
// Reencoded: lwc2
// Reencoded: sdbbp
// Reencoded: sdc2
// Reencoded: swc2
// Removed: /.ps$/, cvt.ps.s, cvt.ps.pw
// Removed: addi
// Removed: bc1any2, bc1any4
// Removed: bc2[ft]
// Removed: bc2f, bc2t
// Removed: bgezal
// Removed: bltzal
// Removed: c.cond.fmt, bc1[ft]
// Removed: div, divu
// Removed: jalx
// Removed: ldxc1
// Removed: luxc1
// Removed: lwl, lwr, lwle, lwre, swl, swr, swle, swre
// Removed: lwxc1
// Removed: madd.[ds], nmadd.[ds], nmsub.[ds], sub.[ds]
// Removed: mfhi, mflo, mthi, mtlo, madd, maddu, msub, msubu, mul
// Removed: movf, movt
// Removed: movf.fmt, movt.fmt, movn.fmt, movz.fmt
// Removed: movn, movz
// Removed: mult, multu
// Removed: prefx
// Removed: sdxc1
// Removed: suxc1
// Removed: swxc1
// Removed: teqi, tgei, tgeiu, tlti, tltiu, tnei
// Rencoded: [ls][wd]c2

//===----------------------------------------------------------------------===//
//
// Instruction Encodings
//
//===----------------------------------------------------------------------===//

class ADDIUPC_ENC : PCREL19_FM<OPCODE2_ADDIUPC>;
class ALIGN_ENC  : SPECIAL3_ALIGN_FM<OPCODE6_ALIGN>;
class ALUIPC_ENC : PCREL16_FM<OPCODE5_ALUIPC>;
class AUI_ENC    : AUI_FM;
class AUIPC_ENC  : PCREL16_FM<OPCODE5_AUIPC>;
class BITSWAP_ENC : SPECIAL3_2R_FM<OPCODE6_BITSWAP>;
class DIV_ENC    : SPECIAL_3R_FM<0b00010, 0b011010>;
class DIVU_ENC   : SPECIAL_3R_FM<0b00010, 0b011011>;
class MOD_ENC    : SPECIAL_3R_FM<0b00011, 0b011010>;
class MODU_ENC   : SPECIAL_3R_FM<0b00011, 0b011011>;
class MUH_ENC    : SPECIAL_3R_FM<0b00011, 0b011000>;
class MUHU_ENC   : SPECIAL_3R_FM<0b00011, 0b011001>;
class MUL_R6_ENC : SPECIAL_3R_FM<0b00010, 0b011000>;
class MULU_ENC   : SPECIAL_3R_FM<0b00010, 0b011001>;
class SEL_D_ENC  : COP1_3R_FM<0b010000, FIELD_FMT_D>;
class SEL_S_ENC  : COP1_3R_FM<0b010000, FIELD_FMT_S>;

class MAX_S_ENC : COP1_3R_FM<0b011101, FIELD_FMT_S>;
class MAX_D_ENC : COP1_3R_FM<0b011101, FIELD_FMT_D>;
class MIN_S_ENC : COP1_3R_FM<0b011100, FIELD_FMT_S>;
class MIN_D_ENC : COP1_3R_FM<0b011100, FIELD_FMT_D>;

class MAXA_S_ENC : COP1_3R_FM<0b011111, FIELD_FMT_S>;
class MAXA_D_ENC : COP1_3R_FM<0b011111, FIELD_FMT_D>;
class MINA_S_ENC : COP1_3R_FM<0b011110, FIELD_FMT_S>;
class MINA_D_ENC : COP1_3R_FM<0b011110, FIELD_FMT_D>;

class SELEQZ_S_ENC : COP1_3R_FM<0b010100, FIELD_FMT_S>;
class SELEQZ_D_ENC : COP1_3R_FM<0b010100, FIELD_FMT_D>;
class SELNEZ_S_ENC : COP1_3R_FM<0b010111, FIELD_FMT_S>;
class SELNEZ_D_ENC : COP1_3R_FM<0b010111, FIELD_FMT_D>;

class RINT_S_ENC : COP1_2R_FM<0b011010, FIELD_FMT_S>;
class RINT_D_ENC : COP1_2R_FM<0b011010, FIELD_FMT_D>;

//===----------------------------------------------------------------------===//
//
// Instruction Descriptions
//
//===----------------------------------------------------------------------===//

class ADDIUPC_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> {
  dag OutOperandList = (outs GPROpnd:$rs);
  dag InOperandList = (ins simm19_lsl2:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rs, $imm");
  list<dag> Pattern = [];
}

class ADDIUPC_DESC : ADDIUPC_DESC_BASE<"addiupc", GPR32Opnd>;

class ALIGN_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                      Operand ImmOpnd> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt, ImmOpnd:$bp);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt, $bp");
  list<dag> Pattern = [];
}

class ALIGN_DESC : ALIGN_DESC_BASE<"align", GPR32Opnd, uimm2>;

class ALUIPC_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> {
  dag OutOperandList = (outs GPROpnd:$rs);
  dag InOperandList = (ins simm16:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rs, $imm");
  list<dag> Pattern = [];
}

class ALUIPC_DESC : ALUIPC_DESC_BASE<"aluipc", GPR32Opnd>;
class AUIPC_DESC : ALUIPC_DESC_BASE<"auipc", GPR32Opnd>;

class AUI_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> {
  dag OutOperandList = (outs GPROpnd:$rs);
  dag InOperandList = (ins GPROpnd:$rt, simm16:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rs, $rt, $imm");
  list<dag> Pattern = [];
}

class AUI_DESC : AUI_DESC_BASE<"aui", GPR32Opnd>;

class BITSWAP_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rt");
  list<dag> Pattern = [];
}

class BITSWAP_DESC : BITSWAP_DESC_BASE<"bitswap", GPR32Opnd>;

class DIVMOD_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt");
  list<dag> Pattern = [];
}

class DIV_DESC  : DIVMOD_DESC_BASE<"div", GPR32Opnd>;
class DIVU_DESC : DIVMOD_DESC_BASE<"divu", GPR32Opnd>;
class MOD_DESC  : DIVMOD_DESC_BASE<"mod", GPR32Opnd>;
class MODU_DESC : DIVMOD_DESC_BASE<"modu", GPR32Opnd>;

class MUL_R6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt");
  list<dag> Pattern = [];
}

class MUH_DESC    : MUL_R6_DESC_BASE<"muh", GPR32Opnd>;
class MUHU_DESC   : MUL_R6_DESC_BASE<"muhu", GPR32Opnd>;
class MUL_R6_DESC : MUL_R6_DESC_BASE<"mul", GPR32Opnd>;
class MULU_DESC   : MUL_R6_DESC_BASE<"mulu", GPR32Opnd>;

class SEL_DESC_BASE<string instr_asm, RegisterOperand FGROpnd> {
  dag OutOperandList = (outs FGROpnd:$fd);
  dag InOperandList = (ins FGROpnd:$fd_in, FGROpnd:$fs, FGROpnd:$ft);
  string AsmString = !strconcat(instr_asm, "\t$fd, $fs, $ft");
  list<dag> Pattern = [];
  string Constraints = "$fd_in = $fd";
}

class SEL_D_DESC : SEL_DESC_BASE<"sel.d", FGR64Opnd>;
class SEL_S_DESC : SEL_DESC_BASE<"sel.s", FGR32Opnd>;

class MAX_MIN_DESC_BASE<string instr_asm, RegisterOperand FGROpnd> {
  dag OutOperandList = (outs FGROpnd:$fd);
  dag InOperandList = (ins FGROpnd:$fs, FGROpnd:$ft);
  string AsmString = !strconcat(instr_asm, "\t$fd, $fs, $ft");
  list<dag> Pattern = [];
}

class MAX_S_DESC : MAX_MIN_DESC_BASE<"max.s", FGR32Opnd>;
class MAX_D_DESC : MAX_MIN_DESC_BASE<"max.d", FGR64Opnd>;
class MIN_S_DESC : MAX_MIN_DESC_BASE<"min.s", FGR32Opnd>;
class MIN_D_DESC : MAX_MIN_DESC_BASE<"min.d", FGR64Opnd>;

class MAXA_S_DESC : MAX_MIN_DESC_BASE<"maxa.s", FGR32Opnd>;
class MAXA_D_DESC : MAX_MIN_DESC_BASE<"maxa.d", FGR64Opnd>;
class MINA_S_DESC : MAX_MIN_DESC_BASE<"mina.s", FGR32Opnd>;
class MINA_D_DESC : MAX_MIN_DESC_BASE<"mina.d", FGR64Opnd>;

class SELEQNEZ_DESC_BASE<string instr_asm, RegisterOperand FGROpnd> {
  dag OutOperandList = (outs FGROpnd:$fd);
  dag InOperandList = (ins FGROpnd:$fs, FGROpnd:$ft);
  string AsmString = !strconcat(instr_asm, "\t$fd, $fs, $ft");
  list<dag> Pattern = [];
}

class SELEQZ_S_DESC : SELEQNEZ_DESC_BASE<"seleqz.s", FGR32Opnd>;
class SELEQZ_D_DESC : SELEQNEZ_DESC_BASE<"seleqz.d", FGR64Opnd>;
class SELNEZ_S_DESC : SELEQNEZ_DESC_BASE<"selnez.s", FGR32Opnd>;
class SELNEZ_D_DESC : SELEQNEZ_DESC_BASE<"selnez.d", FGR64Opnd>;

class RINT_DESC_BASE<string instr_asm, RegisterOperand FGROpnd> {
  dag OutOperandList = (outs FGROpnd:$fd);
  dag InOperandList = (ins FGROpnd:$fs);
  string AsmString = !strconcat(instr_asm, "\t$fd, $fs");
  list<dag> Pattern = [];
}

class RINT_S_DESC : RINT_DESC_BASE<"rint.s", FGR32Opnd>;
class RINT_D_DESC : RINT_DESC_BASE<"rint.d", FGR64Opnd>;

//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

def ADDIUPC : ADDIUPC_ENC, ADDIUPC_DESC, ISA_MIPS32R6;
def ALIGN : ALIGN_ENC, ALIGN_DESC, ISA_MIPS32R6;
def ALUIPC : ALUIPC_ENC, ALUIPC_DESC, ISA_MIPS32R6;
def AUI : AUI_ENC, AUI_DESC, ISA_MIPS32R6;
def AUIPC : AUIPC_ENC, AUIPC_DESC, ISA_MIPS32R6;
def BALC;
def BC1EQZ;
def BC1NEZ;
def BC2EQZ;
def BC2NEZ;
def BC;
def BEQC;
def BEQZALC;
def BEQZC;
def BGEC;  // Also aliased to blec with operands swapped
def BGEUC; // Also aliased to bleuc with operands swapped
def BGEZALC;
def BGEZC;
def BGTZALC;
def BGTZC;
def BITSWAP : BITSWAP_ENC, BITSWAP_DESC, ISA_MIPS32R6;
def BLEZALC;
def BLEZC;
def BLTC; // Also aliased to bgtc with operands swapped
def BLTUC; // Also aliased to bgtuc with operands swapped
def BLTZALC;
def BLTZC;
def BNEC;
def BNEZALC;
def BNEZC;
def BNVC;
def BOVC;
def CLASS_D;
def CLASS_S;
def CMP_CC_D;
def CMP_CC_S;
def DIV : DIV_ENC, DIV_DESC, ISA_MIPS32R6;
def DIVU : DIVU_ENC, DIVU_DESC, ISA_MIPS32R6;
def JIALC;
def JIC;
// def LSA; // See MSA
def LWPC;
def LWUPC;
def MADDF;
def MAXA_D : MAXA_D_ENC, MAXA_D_DESC, ISA_MIPS32R6;
def MAXA_S : MAXA_S_ENC, MAXA_S_DESC, ISA_MIPS32R6;
def MAX_D : MAX_D_ENC, MAX_D_DESC, ISA_MIPS32R6;
def MAX_S : MAX_S_ENC, MAX_S_DESC, ISA_MIPS32R6;
def MINA_D : MINA_D_ENC, MINA_D_DESC, ISA_MIPS32R6;
def MINA_S : MINA_S_ENC, MINA_S_DESC, ISA_MIPS32R6;
def MIN_D : MIN_D_ENC, MIN_D_DESC, ISA_MIPS32R6;
def MIN_S : MIN_S_ENC, MIN_S_DESC, ISA_MIPS32R6;
def MOD : MOD_ENC, MOD_DESC, ISA_MIPS32R6;
def MODU : MODU_ENC, MODU_DESC, ISA_MIPS32R6;
def MSUBF;
def MUH    : MUH_ENC, MUH_DESC, ISA_MIPS32R6;
def MUHU   : MUHU_ENC, MUHU_DESC, ISA_MIPS32R6;
def MUL_R6 : MUL_R6_ENC, MUL_R6_DESC, ISA_MIPS32R6;
def MULU   : MULU_ENC, MULU_DESC, ISA_MIPS32R6;
def NAL; // BAL with rd=0
def RINT_D : RINT_D_ENC, RINT_D_DESC, ISA_MIPS32R6;
def RINT_S : RINT_S_ENC, RINT_S_DESC, ISA_MIPS32R6;
def SELEQZ;
def SELEQZ_D : SELEQZ_D_ENC, SELEQZ_D_DESC, ISA_MIPS32R6;
def SELEQZ_S : SELEQZ_S_ENC, SELEQZ_S_DESC, ISA_MIPS32R6;
def SELNEZ;
def SELNEZ_D : SELNEZ_D_ENC, SELNEZ_D_DESC, ISA_MIPS32R6;
def SELNEZ_S : SELNEZ_S_ENC, SELNEZ_S_DESC, ISA_MIPS32R6;
def SEL_D : SEL_D_ENC, SEL_D_DESC, ISA_MIPS32R6;
def SEL_S : SEL_S_ENC, SEL_S_DESC, ISA_MIPS32R6;
