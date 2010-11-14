//===-- PPCInstPrinter.cpp - Convert PPC MCInst to assembly syntax --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an PPC MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "PPCInstPrinter.h"
#include "llvm/MC/MCInst.h"
//#include "llvm/MC/MCAsmInfo.h"
//#include "llvm/MC/MCExpr.h"
//#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "PPCGenRegisterNames.inc"
#include "PPCGenInstrNames.inc"
using namespace llvm;

#define GET_INSTRUCTION_NAME
#define PPCAsmPrinter PPCInstPrinter
#define MachineInstr MCInst
#include "PPCGenAsmWriter.inc"

StringRef PPCInstPrinter::getOpcodeName(unsigned Opcode) const {
  return getInstructionName(Opcode);
}


void PPCInstPrinter::printInst(const MCInst *MI, raw_ostream &O) {
  // TODO: pseudo ops.
  
  printInstruction(MI, O);
}

