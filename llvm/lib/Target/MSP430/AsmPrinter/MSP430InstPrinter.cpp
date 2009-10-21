//===-- MSP430InstPrinter.cpp - Convert MSP430 MCInst to assembly syntax --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an MSP430 MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "MSP430InstPrinter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "MSP430GenInstrNames.inc"
using namespace llvm;


// Include the auto-generated portion of the assembly writer.
#define MachineInstr MCInst
#define MSP430AsmPrinter MSP430InstPrinter  // FIXME: REMOVE.
#define NO_ASM_WRITER_BOILERPLATE
#include "MSP430GenAsmWriter.inc"
#undef MachineInstr
#undef MSP430AsmPrinter

void MSP430InstPrinter::printInst(const MCInst *MI) {
  printInstruction(MI);
}
