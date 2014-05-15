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

//===----------------------------------------------------------------------===//
//
// Instruction Descriptions
//
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

def ADDIUPC;
def ALIGN; // Known as as BALIGN in DSP ASE
def ALUIPC;
def AUI;
def AUIPC;
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
def BITSWAP; // Known as BITREV in DSP ASE
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
def MAXA_D;
def MAXA_S;
def MAX_D;
def MAX_S;
def MINA_D;
def MINA_S;
def MIN_D;
def MOD : MOD_ENC, MOD_DESC, ISA_MIPS32R6;
def MODU : MODU_ENC, MODU_DESC, ISA_MIPS32R6;
def MSUBF;
def MUH    : MUH_ENC, MUH_DESC, ISA_MIPS32R6;
def MUHU   : MUHU_ENC, MUHU_DESC, ISA_MIPS32R6;
def MUL_R6 : MUL_R6_ENC, MUL_R6_DESC, ISA_MIPS32R6;
def MULU   : MULU_ENC, MULU_DESC, ISA_MIPS32R6;
def NAL; // BAL with rd=0
def RINT_D;
def RINT_S;
def SELEQZ;
def SELEQZ_D;
def SELEQZ_S;
def SELNEZ;
def SELNEZ_D;
def SELNEZ_S;
def SEL_D : SEL_D_ENC, SEL_D_DESC, ISA_MIPS32R6;
def SEL_S : SEL_S_ENC, SEL_S_DESC, ISA_MIPS32R6;
