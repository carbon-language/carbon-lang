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

def brtarget21 : Operand<OtherVT> {
  let EncoderMethod = "getBranchTarget21OpValue";
  let OperandType = "OPERAND_PCREL";
  let DecoderMethod = "DecodeBranchTarget21";
  let ParserMatchClass = MipsJumpTargetAsmOperand;
}

def brtarget26 : Operand<OtherVT> {
  let EncoderMethod = "getBranchTarget26OpValue";
  let OperandType = "OPERAND_PCREL";
  let DecoderMethod = "DecodeBranchTarget26";
  let ParserMatchClass = MipsJumpTargetAsmOperand;
}

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

class BALC_ENC  : BRANCH_OFF26_FM<0b111010>;
class BC_ENC    : BRANCH_OFF26_FM<0b110010>;
class BEQC_ENC  : CMP_BRANCH_OFF16_FM<0b001000>;
class BNEC_ENC  : CMP_BRANCH_OFF16_FM<0b011000>;

class BLTZC_ENC : CMP_BRANCH_OFF16_FM<0b010111>;
class BGEZC_ENC : CMP_BRANCH_OFF16_FM<0b010110>;

class BLEZC_ENC : CMP_BRANCH_RT_OFF16_FM<0b010110>;
class BGTZC_ENC : CMP_BRANCH_RT_OFF16_FM<0b010111>;

class BEQZC_ENC : CMP_BRANCH_OFF21_FM<0b110110>;
class BNEZC_ENC : CMP_BRANCH_OFF21_FM<0b111110>;

class BITSWAP_ENC : SPECIAL3_2R_FM<OPCODE6_BITSWAP>;
class DIV_ENC    : SPECIAL_3R_FM<0b00010, 0b011010>;
class DIVU_ENC   : SPECIAL_3R_FM<0b00010, 0b011011>;
class MOD_ENC    : SPECIAL_3R_FM<0b00011, 0b011010>;
class MODU_ENC   : SPECIAL_3R_FM<0b00011, 0b011011>;
class MUH_ENC    : SPECIAL_3R_FM<0b00011, 0b011000>;
class MUHU_ENC   : SPECIAL_3R_FM<0b00011, 0b011001>;
class MUL_R6_ENC : SPECIAL_3R_FM<0b00010, 0b011000>;
class MULU_ENC   : SPECIAL_3R_FM<0b00010, 0b011001>;

class MADDF_S_ENC  : COP1_3R_FM<0b011000, FIELD_FMT_S>;
class MADDF_D_ENC  : COP1_3R_FM<0b011000, FIELD_FMT_D>;
class MSUBF_S_ENC  : COP1_3R_FM<0b011001, FIELD_FMT_S>;
class MSUBF_D_ENC  : COP1_3R_FM<0b011001, FIELD_FMT_D>;

class SEL_D_ENC  : COP1_3R_FM<0b010000, FIELD_FMT_D>;
class SEL_S_ENC  : COP1_3R_FM<0b010000, FIELD_FMT_S>;

class LWPC_ENC   : PCREL19_FM<OPCODE2_LWPC>;
class LWUPC_ENC  : PCREL19_FM<OPCODE2_LWUPC>;

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
class CLASS_S_ENC : COP1_2R_FM<0b011011, FIELD_FMT_S>;
class CLASS_D_ENC : COP1_2R_FM<0b011011, FIELD_FMT_D>;

class CMP_CONDN_DESC_BASE<string CondStr, string Typestr, RegisterOperand FGROpnd> {
  dag OutOperandList = (outs FGROpnd:$fd);
  dag InOperandList = (ins FGROpnd:$fs, FGROpnd:$ft);
  string AsmString = !strconcat("cmp.", CondStr, ".", Typestr, "\t$fd, $fs, $ft");
  list<dag> Pattern = [];
}

//===----------------------------------------------------------------------===//
//
// Instruction Multiclasses
//
//===----------------------------------------------------------------------===//

multiclass CMP_CC_M <FIELD_CMP_FORMAT Format, string Typestr,
                     RegisterOperand FGROpnd>{
  def CMP_F_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_F>,
                    CMP_CONDN_DESC_BASE<"f", Typestr, FGROpnd>,
                    ISA_MIPS32R6;
  def CMP_UN_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_UN>,
                     CMP_CONDN_DESC_BASE<"un", Typestr, FGROpnd>,
                     ISA_MIPS32R6;
  def CMP_EQ_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_EQ>,
                     CMP_CONDN_DESC_BASE<"eq", Typestr, FGROpnd>,
                     ISA_MIPS32R6;
  def CMP_UEQ_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_UEQ>,
                      CMP_CONDN_DESC_BASE<"ueq", Typestr, FGROpnd>,
                      ISA_MIPS32R6;
  def CMP_OLT_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_OLT>,
                      CMP_CONDN_DESC_BASE<"olt", Typestr, FGROpnd>,
                      ISA_MIPS32R6;
  def CMP_ULT_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_ULT>,
                      CMP_CONDN_DESC_BASE<"ult", Typestr, FGROpnd>,
                      ISA_MIPS32R6;
  def CMP_OLE_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_OLE>,
                      CMP_CONDN_DESC_BASE<"ole", Typestr, FGROpnd>,
                      ISA_MIPS32R6;
  def CMP_ULE_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_ULE>,
                      CMP_CONDN_DESC_BASE<"ule", Typestr, FGROpnd>,
                      ISA_MIPS32R6;
  def CMP_SF_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_SF>,
                     CMP_CONDN_DESC_BASE<"sf", Typestr, FGROpnd>,
                     ISA_MIPS32R6;
  def CMP_NGLE_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_NGLE>,
                       CMP_CONDN_DESC_BASE<"ngle", Typestr, FGROpnd>,
                       ISA_MIPS32R6;
  def CMP_SEQ_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_SEQ>,
                      CMP_CONDN_DESC_BASE<"seq", Typestr, FGROpnd>,
                      ISA_MIPS32R6;
  def CMP_NGL_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_NGL>,
                      CMP_CONDN_DESC_BASE<"ngl", Typestr, FGROpnd>,
                      ISA_MIPS32R6;
  def CMP_LT_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_LT>,
                     CMP_CONDN_DESC_BASE<"lt", Typestr, FGROpnd>,
                     ISA_MIPS32R6;
  def CMP_NGE_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_NGE>,
                      CMP_CONDN_DESC_BASE<"nge", Typestr, FGROpnd>,
                      ISA_MIPS32R6;
  def CMP_LE_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_LE>,
                     CMP_CONDN_DESC_BASE<"le", Typestr, FGROpnd>,
                     ISA_MIPS32R6;
  def CMP_NGT_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_NGT>,
                      CMP_CONDN_DESC_BASE<"ngt", Typestr, FGROpnd>,
                      ISA_MIPS32R6;
}

//===----------------------------------------------------------------------===//
//
// Instruction Descriptions
//
//===----------------------------------------------------------------------===//

class PCREL19_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> {
  dag OutOperandList = (outs GPROpnd:$rs);
  dag InOperandList = (ins simm19_lsl2:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rs, $imm");
  list<dag> Pattern = [];
}

class ADDIUPC_DESC : PCREL19_DESC_BASE<"addiupc", GPR32Opnd>;
class LWPC_DESC: PCREL19_DESC_BASE<"lwpc", GPR32Opnd>;
class LWUPC_DESC: PCREL19_DESC_BASE<"lwupc", GPR32Opnd>;

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

class BRANCH_DESC_BASE {
  bit isBranch = 1;
  bit isTerminator = 1;
  bit hasDelaySlot = 0;
}

class BC_DESC_BASE<string instr_asm, DAGOperand opnd> : BRANCH_DESC_BASE {
  dag InOperandList = (ins opnd:$offset);
  dag OutOperandList = (outs);
  string AsmString = !strconcat(instr_asm, "\t$offset");
  bit isBarrier = 1;
}

class CMP_BC_DESC_BASE<string instr_asm, DAGOperand opnd,
                       RegisterOperand GPROpnd> : BRANCH_DESC_BASE {
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt, opnd:$offset);
  dag OutOperandList = (outs);
  string AsmString = !strconcat(instr_asm, "\t$rs, $rt, $offset");
  list<Register> Defs = [AT];
}

class CMP_CBR_EQNE_Z_DESC_BASE<string instr_asm, DAGOperand opnd,
                               RegisterOperand GPROpnd> : BRANCH_DESC_BASE {
  dag InOperandList = (ins GPROpnd:$rs, opnd:$offset);
  dag OutOperandList = (outs);
  string AsmString = !strconcat(instr_asm, "\t$rs, $offset");
  list<Register> Defs = [AT];
}

class CMP_CBR_RT_Z_DESC_BASE<string instr_asm, DAGOperand opnd,
                             RegisterOperand GPROpnd> : BRANCH_DESC_BASE {
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt, opnd:$offset);
  dag OutOperandList = (outs);
  string AsmString = !strconcat(instr_asm, "\t$rt, $offset");
  list<Register> Defs = [AT];
}

class BALC_DESC : BC_DESC_BASE<"balc", brtarget26> {
  bit isCall = 1;
  list<Register> Defs = [RA];
}

class BC_DESC : BC_DESC_BASE<"bc", brtarget26>;
class BEQC_DESC : CMP_BC_DESC_BASE<"beqc", brtarget, GPR32Opnd>;
class BNEC_DESC : CMP_BC_DESC_BASE<"bnec", brtarget, GPR32Opnd>;

class BLTZC_DESC : CMP_CBR_RT_Z_DESC_BASE<"bltzc", brtarget, GPR32Opnd> {
  string Constraints = "$rs = $rt";
}

class BGEZC_DESC : CMP_CBR_RT_Z_DESC_BASE<"bgezc", brtarget, GPR32Opnd> {
  string Constraints = "$rs = $rt";
}

class BLEZC_DESC : CMP_CBR_RT_Z_DESC_BASE<"blezc", brtarget, GPR32Opnd>;
class BGTZC_DESC : CMP_CBR_RT_Z_DESC_BASE<"bgtzc", brtarget, GPR32Opnd>;

class BEQZC_DESC : CMP_CBR_EQNE_Z_DESC_BASE<"beqzc", brtarget21, GPR32Opnd>;
class BNEZC_DESC : CMP_CBR_EQNE_Z_DESC_BASE<"bnezc", brtarget21, GPR32Opnd>;

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

class COP1_4R_DESC_BASE<string instr_asm, RegisterOperand FGROpnd> {
  dag OutOperandList = (outs FGROpnd:$fd);
  dag InOperandList = (ins FGROpnd:$fd_in, FGROpnd:$fs, FGROpnd:$ft);
  string AsmString = !strconcat(instr_asm, "\t$fd, $fs, $ft");
  list<dag> Pattern = [];
  string Constraints = "$fd_in = $fd";
}

class SEL_D_DESC : COP1_4R_DESC_BASE<"sel.d", FGR64Opnd>;
class SEL_S_DESC : COP1_4R_DESC_BASE<"sel.s", FGR32Opnd>;

class MADDF_S_DESC  : COP1_4R_DESC_BASE<"maddf.s", FGR32Opnd>;
class MADDF_D_DESC  : COP1_4R_DESC_BASE<"maddf.d", FGR64Opnd>;
class MSUBF_S_DESC  : COP1_4R_DESC_BASE<"msubf.s", FGR32Opnd>;
class MSUBF_D_DESC  : COP1_4R_DESC_BASE<"msubf.d", FGR64Opnd>;

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

class CLASS_RINT_DESC_BASE<string instr_asm, RegisterOperand FGROpnd> {
  dag OutOperandList = (outs FGROpnd:$fd);
  dag InOperandList = (ins FGROpnd:$fs);
  string AsmString = !strconcat(instr_asm, "\t$fd, $fs");
  list<dag> Pattern = [];
}

class RINT_S_DESC : CLASS_RINT_DESC_BASE<"rint.s", FGR32Opnd>;
class RINT_D_DESC : CLASS_RINT_DESC_BASE<"rint.d", FGR64Opnd>;
class CLASS_S_DESC : CLASS_RINT_DESC_BASE<"class.s", FGR32Opnd>;
class CLASS_D_DESC : CLASS_RINT_DESC_BASE<"class.d", FGR64Opnd>;

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
def BALC : BALC_ENC, BALC_DESC, ISA_MIPS32R6;
def BC1EQZ;
def BC1NEZ;
def BC2EQZ;
def BC2NEZ;
def BC : BC_ENC, BC_DESC, ISA_MIPS32R6;
def BEQC : BEQC_ENC, BEQC_DESC, ISA_MIPS32R6;
def BEQZALC;
def BEQZC : BEQZC_ENC, BEQZC_DESC, ISA_MIPS32R6;
def BGEC;  // Also aliased to blec with operands swapped
def BGEUC; // Also aliased to bleuc with operands swapped
def BGEZALC;
def BGEZC : BGEZC_ENC, BGEZC_DESC, ISA_MIPS32R6;
def BGTZALC;
def BGTZC : BGTZC_ENC, BGTZC_DESC, ISA_MIPS32R6;
def BITSWAP : BITSWAP_ENC, BITSWAP_DESC, ISA_MIPS32R6;
def BLEZALC;
def BLEZC : BLEZC_ENC, BLEZC_DESC, ISA_MIPS32R6;
def BLTC; // Also aliased to bgtc with operands swapped
def BLTUC; // Also aliased to bgtuc with operands swapped
def BLTZALC;
def BLTZC : BLTZC_ENC, BLTZC_DESC, ISA_MIPS32R6;
def BNEC : BNEC_ENC, BNEC_DESC, ISA_MIPS32R6;
def BNEZALC;
def BNEZC : BNEZC_ENC, BNEZC_DESC, ISA_MIPS32R6;
def BNVC;
def BOVC;
def CLASS_D : CLASS_D_ENC, CLASS_D_DESC, ISA_MIPS32R6;
def CLASS_S : CLASS_S_ENC, CLASS_S_DESC, ISA_MIPS32R6;
defm S : CMP_CC_M<FIELD_CMP_FORMAT_S, "s", FGR32Opnd>;
defm D : CMP_CC_M<FIELD_CMP_FORMAT_D, "d", FGR64Opnd>;
def DIV : DIV_ENC, DIV_DESC, ISA_MIPS32R6;
def DIVU : DIVU_ENC, DIVU_DESC, ISA_MIPS32R6;
def JIALC;
def JIC;
// def LSA; // See MSA
def LWPC : LWPC_ENC, LWPC_DESC, ISA_MIPS32R6;
def LWUPC : LWUPC_ENC, LWUPC_DESC, ISA_MIPS32R6;
def MADDF_S : MADDF_S_ENC, MADDF_S_DESC, ISA_MIPS32R6;
def MADDF_D : MADDF_D_ENC, MADDF_D_DESC, ISA_MIPS32R6;
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
def MSUBF_S : MSUBF_S_ENC, MSUBF_S_DESC, ISA_MIPS32R6;
def MSUBF_D : MSUBF_D_ENC, MSUBF_D_DESC, ISA_MIPS32R6;
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
