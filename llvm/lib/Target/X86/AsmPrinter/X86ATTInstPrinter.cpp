//===-- X86ATTInstPrinter.cpp - AT&T assembly instruction printing --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file includes code for rendering MCInst instances as AT&T-style
// assembly.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "llvm/MC/MCInst.h"
#include "X86ATTAsmPrinter.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// Include the auto-generated portion of the assembly writer.
#define MachineInstr MCInst
#define NO_ASM_WRITER_BOILERPLATE
#include "X86GenAsmWriter.inc"
#undef MachineInstr

void X86ATTAsmPrinter::printSSECC(const MCInst *MI, unsigned Op) {
  unsigned char value = MI->getOperand(Op).getImm();
  assert(value <= 7 && "Invalid ssecc argument!");
  switch (value) {
    case 0: O << "eq"; break;
    case 1: O << "lt"; break;
    case 2: O << "le"; break;
    case 3: O << "unord"; break;
    case 4: O << "neq"; break;
    case 5: O << "nlt"; break;
    case 6: O << "nle"; break;
    case 7: O << "ord"; break;
  }
}


void X86ATTAsmPrinter::printPICLabel(const MCInst *MI, unsigned Op) {
  assert(0 &&
         "This is only used for MOVPC32r, should lower before asm printing!");
}


void X86ATTAsmPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                    const char *Modifier, bool NotRIPRel) {
}

void X86ATTAsmPrinter::printLeaMemReference(const MCInst *MI, unsigned Op,
                                            const char *Modifier,
                                            bool NotRIPRel) {
}

void X86ATTAsmPrinter::printMemReference(const MCInst *MI, unsigned Op,
                                         const char *Modifier, bool NotRIPRel){
}
