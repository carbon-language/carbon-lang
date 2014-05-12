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

class DMUH_ENC    : SPECIAL_3R_FM<0b00011, 0b111000>;
class DMUHU_ENC   : SPECIAL_3R_FM<0b00011, 0b111001>;
class DMUL_R6_ENC : SPECIAL_3R_FM<0b00010, 0b111000>;
class DMULU_ENC   : SPECIAL_3R_FM<0b00010, 0b111001>;

//===----------------------------------------------------------------------===//
//
// Instruction Descriptions
//
//===----------------------------------------------------------------------===//

class DMUH_DESC    : MUL_R6_DESC_BASE<"dmuh", GPR64Opnd>;
class DMUHU_DESC   : MUL_R6_DESC_BASE<"dmuhu", GPR64Opnd>;
class DMUL_R6_DESC : MUL_R6_DESC_BASE<"dmul", GPR64Opnd>;
class DMULU_DESC   : MUL_R6_DESC_BASE<"dmulu", GPR64Opnd>;

//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

def DAHI;
def DALIGN;
def DATI;
def DAUI;
def DBITSWAP;
def DDIV;
def DDIVU;
// def DLSA; // See MSA
def DMOD;
def DMODU;
def DMUH: DMUH_ENC, DMUH_DESC, ISA_MIPS64R6;
def DMUHU: DMUHU_ENC, DMUHU_DESC, ISA_MIPS64R6;
def DMUL_R6: DMUL_R6_ENC, DMUL_R6_DESC, ISA_MIPS64R6;
def DMULU: DMULU_ENC, DMULU_DESC, ISA_MIPS64R6;
def LDPC;
