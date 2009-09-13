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
#include "llvm/Target/TargetRegisterInfo.h"  // FIXME: REMOVE.
using namespace llvm;

// Include the auto-generated portion of the assembly writer.
#define MachineInstr MCInst
#define NO_ASM_WRITER_BOILERPLATE
#include "X86GenAsmWriter.inc"
#undef MachineInstr

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


void X86ATTInstPrinter::printPICLabel(const MCInst *MI, unsigned Op) {
  llvm_unreachable("This is only used for MOVPC32r,"
                   "should lower before instruction printing!");
}


/// print_pcrel_imm - This is used to print an immediate value that ends up
/// being encoded as a pc-relative value.  These print slightly differently, for
/// example, a $ is not emitted.
void X86ATTInstPrinter::print_pcrel_imm(const MCInst *MI, unsigned OpNo) {
  const MCOperand &Op = MI->getOperand(OpNo);
  
  if (Op.isImm())
    O << Op.getImm();
  else if (Op.isExpr())
    Op.getExpr()->print(O, MAI);
  else
    llvm_unreachable("Unknown pcrel immediate operand");
}


void X86ATTInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                    const char *Modifier) {
  assert(Modifier == 0 && "Modifiers should not be used");
  
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    O << '%' << TRI->getAsmName(Op.getReg());
    return;
  } else if (Op.isImm()) {
    O << '$' << Op.getImm();
    return;
  } else if (Op.isExpr()) {
    O << '$';
    Op.getExpr()->print(O, MAI);
    return;
  }
  
  O << "<<UNKNOWN OPERAND KIND>>";
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
    DispSpec.getExpr()->print(O, MAI);
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
  const MCOperand &Segment = MI->getOperand(Op+4);
  if (Segment.getReg()) {
    printOperand(MI, Op+4);
    O << ':';
  }
  printLeaMemReference(MI, Op);
}
