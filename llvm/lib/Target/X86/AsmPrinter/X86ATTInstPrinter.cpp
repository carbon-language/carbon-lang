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
#include "X86ATTInstPrinter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "X86GenInstrNames.inc"
using namespace llvm;

// Include the auto-generated portion of the assembly writer.
#define MachineInstr MCInst
#define NO_ASM_WRITER_BOILERPLATE
#include "X86GenAsmWriter.inc"
#undef MachineInstr

void X86ATTInstPrinter::printInst(const MCInst *MI) { printInstruction(MI); }

void X86ATTInstPrinter::printSSECC(const MCInst *MI, unsigned Op) {
  switch (MI->getOperand(Op).getImm()) {
  default: llvm_unreachable("Invalid ssecc argument!");
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

/// print_pcrel_imm - This is used to print an immediate value that ends up
/// being encoded as a pc-relative value (e.g. for jumps and calls).  These
/// print slightly differently than normal immediates.  For example, a $ is not
/// emitted.
void X86ATTInstPrinter::print_pcrel_imm(const MCInst *MI, unsigned OpNo) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm())
    // Print this as a signed 32-bit value.
    O << (int)Op.getImm();
  else {
    assert(Op.isExpr() && "unknown pcrel immediate operand");
    Op.getExpr()->print(O, &MAI);
  }
}

void X86ATTInstPrinter::printOperand(const MCInst *MI, unsigned OpNo) {
  
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    O << '%' << getRegisterName(Op.getReg());
  } else if (Op.isImm()) {
    O << '$' << Op.getImm();
  } else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    O << '$';
    Op.getExpr()->print(O, &MAI);
  }
}

void X86ATTInstPrinter::printLeaMemReference(const MCInst *MI, unsigned Op) {
  const MCOperand &BaseReg  = MI->getOperand(Op);
  const MCOperand &IndexReg = MI->getOperand(Op+2);
  const MCOperand &DispSpec = MI->getOperand(Op+3);
  
  if (DispSpec.isImm()) {
    int64_t DispVal = DispSpec.getImm();
    if (DispVal || (!IndexReg.getReg() && !BaseReg.getReg()))
      O << DispVal;
  } else {
    assert(DispSpec.isExpr() && "non-immediate displacement for LEA?");
    DispSpec.getExpr()->print(O, &MAI);
  }
  
  if (IndexReg.getReg() || BaseReg.getReg()) {
    O << '(';
    if (BaseReg.getReg())
      printOperand(MI, Op);
    
    if (IndexReg.getReg()) {
      O << ',';
      printOperand(MI, Op+2);
      unsigned ScaleVal = MI->getOperand(Op+1).getImm();
      if (ScaleVal != 1)
        O << ',' << ScaleVal;
    }
    O << ')';
  }
}

void X86ATTInstPrinter::printMemReference(const MCInst *MI, unsigned Op) {
  // If this has a segment register, print it.
  if (MI->getOperand(Op+4).getReg()) {
    printOperand(MI, Op+4);
    O << ':';
  }
  printLeaMemReference(MI, Op);
}
