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
def DMUH;
def DMUHU;
def DMUL_R6; // Not to be confused with the old mul
def DMULU;
def LDPC;
