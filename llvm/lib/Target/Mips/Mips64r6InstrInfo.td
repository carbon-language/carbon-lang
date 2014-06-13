//=- Mips64r6InstrInfo.td - Mips64r6 Instruction Information -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips64r6 instructions.
//
//===----------------------------------------------------------------------===//

// Notes about removals/changes from MIPS32r6:
// Reencoded: dclo, dclz
// Reencoded: lld, scd

//===----------------------------------------------------------------------===//
//
// Instruction Encodings
//
//===----------------------------------------------------------------------===//

class DALIGN_ENC  : SPECIAL3_DALIGN_FM<OPCODE6_DALIGN>;
class DAUI_ENC    : DAUI_FM;
class DAHI_ENC    : REGIMM_FM<OPCODE5_DAHI>;
class DATI_ENC    : REGIMM_FM<OPCODE5_DATI>;
class DBITSWAP_ENC : SPECIAL3_2R_FM<OPCODE6_DBITSWAP>;
class DDIV_ENC    : SPECIAL_3R_FM<0b00010, 0b011110>;
class DDIVU_ENC   : SPECIAL_3R_FM<0b00010, 0b011111>;
class DMOD_ENC    : SPECIAL_3R_FM<0b00011, 0b011110>;
class DMODU_ENC   : SPECIAL_3R_FM<0b00011, 0b011111>;
class DMUH_ENC    : SPECIAL_3R_FM<0b00011, 0b111000>;
class DMUHU_ENC   : SPECIAL_3R_FM<0b00011, 0b111001>;
class DMUL_R6_ENC : SPECIAL_3R_FM<0b00010, 0b111000>;
class DMULU_ENC   : SPECIAL_3R_FM<0b00010, 0b111001>;
class LDPC_ENC    : PCREL18_FM<OPCODE3_LDPC>;

//===----------------------------------------------------------------------===//
//
// Instruction Descriptions
//
//===----------------------------------------------------------------------===//

class AHI_ATI_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> {
  dag OutOperandList = (outs GPROpnd:$rs);
  dag InOperandList = (ins GPROpnd:$rt, simm16:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rt, $imm");
  string Constraints = "$rs = $rt";
}

class DALIGN_DESC  : ALIGN_DESC_BASE<"dalign", GPR64Opnd, uimm3>;
class DAHI_DESC    : AHI_ATI_DESC_BASE<"dahi", GPR64Opnd>;
class DATI_DESC    : AHI_ATI_DESC_BASE<"dati", GPR64Opnd>;
class DAUI_DESC    : AUI_DESC_BASE<"daui", GPR64Opnd>;
class DBITSWAP_DESC : BITSWAP_DESC_BASE<"dbitswap", GPR64Opnd>;
class DDIV_DESC    : DIVMOD_DESC_BASE<"ddiv", GPR64Opnd, sdiv>;
class DDIVU_DESC   : DIVMOD_DESC_BASE<"ddivu", GPR64Opnd, udiv>;
class DMOD_DESC    : DIVMOD_DESC_BASE<"dmod", GPR64Opnd, srem>;
class DMODU_DESC   : DIVMOD_DESC_BASE<"dmodu", GPR64Opnd, urem>;
class DMUH_DESC    : MUL_R6_DESC_BASE<"dmuh", GPR64Opnd, mulhs>;
class DMUHU_DESC   : MUL_R6_DESC_BASE<"dmuhu", GPR64Opnd, mulhu>;
class DMUL_R6_DESC : MUL_R6_DESC_BASE<"dmul", GPR64Opnd, mul>;
class DMULU_DESC   : MUL_R6_DESC_BASE<"dmulu", GPR64Opnd>;
class LDPC_DESC    : PCREL_DESC_BASE<"ldpc", GPR64Opnd, simm18_lsl3>;

class SELEQZ64_DESC : SELEQNE_Z_DESC_BASE<"seleqz", GPR64Opnd>;
class SELNEZ64_DESC : SELEQNE_Z_DESC_BASE<"selnez", GPR64Opnd>;

//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

def DAHI : DAHI_ENC, DAHI_DESC, ISA_MIPS64R6;
def DALIGN : DALIGN_ENC, DALIGN_DESC, ISA_MIPS64R6;
def DATI : DATI_ENC, DATI_DESC, ISA_MIPS64R6;
def DAUI : DAUI_ENC, DAUI_DESC, ISA_MIPS64R6;
def DBITSWAP : DBITSWAP_ENC, DBITSWAP_DESC, ISA_MIPS64R6;
def DDIV : DDIV_ENC, DDIV_DESC, ISA_MIPS64R6;
def DDIVU : DDIVU_ENC, DDIVU_DESC, ISA_MIPS64R6;
// def DLSA; // See MSA
def DMOD : DMOD_ENC, DMOD_DESC, ISA_MIPS64R6;
def DMODU : DMODU_ENC, DMODU_DESC, ISA_MIPS64R6;
def DMUH: DMUH_ENC, DMUH_DESC, ISA_MIPS64R6;
def DMUHU: DMUHU_ENC, DMUHU_DESC, ISA_MIPS64R6;
def DMUL_R6: DMUL_R6_ENC, DMUL_R6_DESC, ISA_MIPS64R6;
def DMULU: DMULU_ENC, DMULU_DESC, ISA_MIPS64R6;
def LDPC: LDPC_ENC, LDPC_DESC, ISA_MIPS64R6;
let DecoderNamespace = "Mips32r6_64r6_GP64" in {
  def SELEQZ64 : SELEQZ_ENC, SELEQZ64_DESC, ISA_MIPS32R6, GPR_64;
  def SELNEZ64 : SELNEZ_ENC, SELNEZ64_DESC, ISA_MIPS32R6, GPR_64;
}

//===----------------------------------------------------------------------===//
//
// Patterns and Pseudo Instructions
//
//===----------------------------------------------------------------------===//

// i64 selects
def : MipsPat<(select i64:$cond, i64:$t, i64:$f),
              (OR64 (SELNEZ64 i64:$t, i64:$cond),
                    (SELEQZ64 i64:$f, i64:$cond))>,
              ISA_MIPS64R6;
def : MipsPat<(select (i32 (seteq i64:$cond, immz)), i64:$t, i64:$f),
              (OR64 (SELNEZ64 i64:$t, i64:$cond),
                    (SELEQZ64 i64:$f, i64:$cond))>,
              ISA_MIPS64R6;
def : MipsPat<(select (i32 (setne i64:$cond, immz)), i64:$t, i64:$f),
              (OR64 (SELNEZ64 i64:$f, i64:$cond),
                    (SELEQZ64 i64:$t, i64:$cond))>,
              ISA_MIPS64R6;
def : MipsPat<(select (i32 (seteq i64:$cond, immZExt16_64:$imm)), i64:$t, i64:$f),
              (OR64 (SELNEZ64 i64:$t, (XORi64 i64:$cond, immZExt16_64:$imm)),
                    (SELEQZ64 i64:$f, (XORi64 i64:$cond, immZExt16_64:$imm)))>,
              ISA_MIPS64R6;
def : MipsPat<(select (i32 (setne i64:$cond, immZExt16_64:$imm)), i64:$t, i64:$f),
              (OR64 (SELNEZ64 i64:$f, (XORi64 i64:$cond, immZExt16_64:$imm)),
                    (SELEQZ64 i64:$t, (XORi64 i64:$cond, immZExt16_64:$imm)))>,
              ISA_MIPS64R6;
def : MipsPat<
  (select (i32 (setgt i64:$cond, immSExt16Plus1:$imm)), i64:$t, i64:$f),
  (OR64 (SELNEZ64 i64:$t,
                  (SUBREG_TO_REG (i64 0), (SLTi64 i64:$cond, (Plus1 imm:$imm)),
                                 sub_32)),
        (SELEQZ64 i64:$f,
                  (SUBREG_TO_REG (i64 0), (SLTi64 i64:$cond, (Plus1 imm:$imm)),
                                 sub_32)))>,
  ISA_MIPS64R6;
def : MipsPat<
  (select (i32 (setugt i64:$cond, immSExt16Plus1:$imm)), i64:$t, i64:$f),
  (OR64 (SELNEZ64 i64:$t,
                  (SUBREG_TO_REG (i64 0), (SLTiu64 i64:$cond, (Plus1 imm:$imm)),
                                 sub_32)),
        (SELEQZ64 i64:$f,
                  (SUBREG_TO_REG (i64 0), (SLTiu64 i64:$cond, (Plus1 imm:$imm)),
                                 sub_32)))>,
  ISA_MIPS64R6;

def : MipsPat<(select (i32 (setne i64:$cond, immz)), i64:$t, immz),
              (SELNEZ64 i64:$t, i64:$cond)>, ISA_MIPS64R6;
def : MipsPat<(select (i32 (seteq i64:$cond, immz)), i64:$t, immz),
              (SELEQZ64 i64:$t, i64:$cond)>, ISA_MIPS64R6;
def : MipsPat<(select (i32 (setne i64:$cond, immz)), immz, i64:$f),
              (SELEQZ64 i64:$f, i64:$cond)>, ISA_MIPS64R6;
def : MipsPat<(select (i32 (seteq i64:$cond, immz)), immz, i64:$f),
              (SELNEZ64 i64:$f, i64:$cond)>, ISA_MIPS64R6;

// i64 selects from an i32 comparison
// One complicating factor here is that bits 32-63 of an i32 are undefined.
// FIXME: Ideally, setcc would always produce an i64 on MIPS64 targets.
//        This would allow us to remove the sign-extensions here.
def : MipsPat<(select i32:$cond, i64:$t, i64:$f),
              (OR64 (SELNEZ64 i64:$t, (SLL64_32 i32:$cond)),
                    (SELEQZ64 i64:$f, (SLL64_32 i32:$cond)))>,
              ISA_MIPS64R6;
def : MipsPat<(select (i32 (seteq i32:$cond, immz)), i64:$t, i64:$f),
              (OR64 (SELNEZ64 i64:$t, (SLL64_32 i32:$cond)),
                    (SELEQZ64 i64:$f, (SLL64_32 i32:$cond)))>,
              ISA_MIPS64R6;
def : MipsPat<(select (i32 (setne i32:$cond, immz)), i64:$t, i64:$f),
              (OR64 (SELNEZ64 i64:$f, (SLL64_32 i32:$cond)),
                    (SELEQZ64 i64:$t, (SLL64_32 i32:$cond)))>,
              ISA_MIPS64R6;
def : MipsPat<(select (i32 (seteq i32:$cond, immZExt16:$imm)), i64:$t, i64:$f),
              (OR64 (SELNEZ64 i64:$t, (SLL64_32 (XORi i32:$cond,
                                                      immZExt16:$imm))),
                    (SELEQZ64 i64:$f, (SLL64_32 (XORi i32:$cond,
                                                      immZExt16:$imm))))>,
              ISA_MIPS64R6;
def : MipsPat<(select (i32 (setne i32:$cond, immZExt16:$imm)), i64:$t, i64:$f),
              (OR64 (SELNEZ64 i64:$f, (SLL64_32 (XORi i32:$cond,
                                                      immZExt16:$imm))),
                    (SELEQZ64 i64:$t, (SLL64_32 (XORi i32:$cond,
                                                      immZExt16:$imm))))>,
              ISA_MIPS64R6;

def : MipsPat<(select i32:$cond, i64:$t, immz),
              (SELNEZ64 i64:$t, (SLL64_32 i32:$cond))>,
              ISA_MIPS64R6;
def : MipsPat<(select (i32 (setne i32:$cond, immz)), i64:$t, immz),
              (SELNEZ64 i64:$t, (SLL64_32 i32:$cond))>,
              ISA_MIPS64R6;
def : MipsPat<(select (i32 (seteq i32:$cond, immz)), i64:$t, immz),
              (SELEQZ64 i64:$t, (SLL64_32 i32:$cond))>,
              ISA_MIPS64R6;
def : MipsPat<(select i32:$cond, immz, i64:$f),
              (SELEQZ64 i64:$f, (SLL64_32 i32:$cond))>,
              ISA_MIPS64R6;
def : MipsPat<(select (i32 (setne i32:$cond, immz)), immz, i64:$f),
              (SELEQZ64 i64:$f, (SLL64_32 i32:$cond))>,
              ISA_MIPS64R6;
def : MipsPat<(select (i32 (seteq i32:$cond, immz)), immz, i64:$f),
              (SELNEZ64 i64:$f, (SLL64_32 i32:$cond))>,
              ISA_MIPS64R6;
