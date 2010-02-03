//===-- X86IntelInstPrinter.cpp - AT&T assembly instruction printing ------===//
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
#include "X86IntelInstPrinter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "X86GenInstrNames.inc"
using namespace llvm;

// Include the auto-generated portion of the assembly writer.
#define MachineInstr MCInst
#include "X86GenAsmWriter1.inc"
#undef MachineInstr

void X86IntelInstPrinter::printInst(const MCInst *MI) { printInstruction(MI); }

void X86IntelInstPrinter::printSSECC(const MCInst *MI, unsigned Op) {
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
/// being encoded as a pc-relative value.
void X86IntelInstPrinter::print_pcrel_imm(const MCInst *MI, unsigned OpNo) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm())
    O << Op.getImm();
  else {
    assert(Op.isExpr() && "unknown pcrel immediate operand");
    O << *Op.getExpr();
  }
}

static void PrintRegName(raw_ostream &O, StringRef RegName) {
  for (unsigned i = 0, e = RegName.size(); i != e; ++i)
    O << (char)toupper(RegName[i]);
}

void X86IntelInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                     const char *Modifier) {
  assert(Modifier == 0 && "Modifiers should not be used");
  
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    PrintRegName(O, getRegisterName(Op.getReg()));
  } else if (Op.isImm()) {
    O << Op.getImm();
  } else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    O << *Op.getExpr();
  }
}

void X86IntelInstPrinter::printLeaMemReference(const MCInst *MI, unsigned Op) {
  const MCOperand &BaseReg  = MI->getOperand(Op);
  unsigned ScaleVal         = MI->getOperand(Op+1).getImm();
  const MCOperand &IndexReg = MI->getOperand(Op+2);
  const MCOperand &DispSpec = MI->getOperand(Op+3);
  
  O << '[';
  
  bool NeedPlus = false;
  if (BaseReg.getReg()) {
    printOperand(MI, Op);
    NeedPlus = true;
  }
  
  if (IndexReg.getReg()) {
    if (NeedPlus) O << " + ";
    if (ScaleVal != 1)
      O << ScaleVal << '*';
    printOperand(MI, Op+2);
    NeedPlus = true;
  }
  
 
  if (!DispSpec.isImm()) {
    if (NeedPlus) O << " + ";
    assert(DispSpec.isExpr() && "non-immediate displacement for LEA?");
    O << *DispSpec.getExpr();
  } else {
    int64_t DispVal = DispSpec.getImm();
    if (DispVal || (!IndexReg.getReg() && !BaseReg.getReg())) {
      if (NeedPlus) {
        if (DispVal > 0)
          O << " + ";
        else {
          O << " - ";
          DispVal = -DispVal;
        }
      }
      O << DispVal;
    }
  }
  
  O << ']';
}

void X86IntelInstPrinter::printMemReference(const MCInst *MI, unsigned Op) {
  // If this has a segment register, print it.
  if (MI->getOperand(Op+4).getReg()) {
    printOperand(MI, Op+4);
    O << ':';
  }
  printLeaMemReference(MI, Op);
}
