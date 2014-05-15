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
// Removed: daddi
// Removed: ddiv, ddivu, dmult, dmultu
// Removed: div, divu
// Removed: ldl, ldr, ldle, ldre, sdl, sdr, sdle, sdre

//===----------------------------------------------------------------------===//
//
// Instruction Encodings
//
//===----------------------------------------------------------------------===//

class DAUI_ENC    : DAUI_FM;
class DAHI_ENC    : REGIMM_FM<OPCODE5_DAHI>;
class DATI_ENC    : REGIMM_FM<OPCODE5_DATI>;
class DDIV_ENC    : SPECIAL_3R_FM<0b00010, 0b011110>;
class DDIVU_ENC   : SPECIAL_3R_FM<0b00010, 0b011111>;
class DMOD_ENC    : SPECIAL_3R_FM<0b00011, 0b011110>;
class DMODU_ENC   : SPECIAL_3R_FM<0b00011, 0b011111>;
class DMUH_ENC    : SPECIAL_3R_FM<0b00011, 0b111000>;
class DMUHU_ENC   : SPECIAL_3R_FM<0b00011, 0b111001>;
class DMUL_R6_ENC : SPECIAL_3R_FM<0b00010, 0b111000>;
class DMULU_ENC   : SPECIAL_3R_FM<0b00010, 0b111001>;

//===----------------------------------------------------------------------===//
//
// Instruction Descriptions
//
//===----------------------------------------------------------------------===//

class DAHI_DESC    : AUI_DESC_BASE<"dahi", GPR64Opnd>;
class DATI_DESC    : AUI_DESC_BASE<"dati", GPR64Opnd>;
class DAUI_DESC    : AUI_DESC_BASE<"daui", GPR64Opnd>;
class DDIV_DESC    : DIVMOD_DESC_BASE<"ddiv", GPR64Opnd>;
class DDIVU_DESC   : DIVMOD_DESC_BASE<"ddivu", GPR64Opnd>;
class DMOD_DESC    : DIVMOD_DESC_BASE<"dmod", GPR64Opnd>;
class DMODU_DESC   : DIVMOD_DESC_BASE<"dmodu", GPR64Opnd>;
class DMUH_DESC    : MUL_R6_DESC_BASE<"dmuh", GPR64Opnd>;
class DMUHU_DESC   : MUL_R6_DESC_BASE<"dmuhu", GPR64Opnd>;
class DMUL_R6_DESC : MUL_R6_DESC_BASE<"dmul", GPR64Opnd>;
class DMULU_DESC   : MUL_R6_DESC_BASE<"dmulu", GPR64Opnd>;

//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

def DAHI : DAHI_ENC, DAHI_DESC, ISA_MIPS64R6;
def DALIGN;
def DATI : DATI_ENC, DATI_DESC, ISA_MIPS64R6;
def DAUI : DAUI_ENC, DAUI_DESC, ISA_MIPS64R6;
def DBITSWAP;
def DDIV : DDIV_ENC, DDIV_DESC, ISA_MIPS64R6;
def DDIVU : DDIVU_ENC, DDIVU_DESC, ISA_MIPS64R6;
// def DLSA; // See MSA
def DMOD : DMOD_ENC, DMOD_DESC, ISA_MIPS64R6;
def DMODU : DMODU_ENC, DMODU_DESC, ISA_MIPS64R6;
def DMUH: DMUH_ENC, DMUH_DESC, ISA_MIPS64R6;
def DMUHU: DMUHU_ENC, DMUHU_DESC, ISA_MIPS64R6;
def DMUL_R6: DMUL_R6_ENC, DMUL_R6_DESC, ISA_MIPS64R6;
def DMULU: DMULU_ENC, DMULU_DESC, ISA_MIPS64R6;
def LDPC;
