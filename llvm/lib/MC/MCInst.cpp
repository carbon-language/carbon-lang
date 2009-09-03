//===- lib/MC/MCInst.cpp - MCInst implementation --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void MCOperand::print(raw_ostream &OS, const MCAsmInfo *MAI) const {
  OS << "<MCOperand ";
  if (!isValid())
    OS << "INVALID";
  else if (isReg())
    OS << "Reg:" << getReg();
  else if (isImm())
    OS << "Imm:" << getImm();
  else if (isMBBLabel())
    OS << "MBB:(" << getMBBLabelFunction() << ","
       << getMBBLabelBlock() << ")";
  else if (isExpr()) {
    OS << "Expr:(";
    getExpr()->print(OS, MAI);
    OS << ")";
  } else
    OS << "UNDEFINED";
  OS << ">";
}

void MCOperand::dump() const {
  print(errs(), 0);
  errs() << "\n";
}

void MCInst::print(raw_ostream &OS, const MCAsmInfo *MAI) const {
  OS << "<MCInst " << getOpcode();
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    OS << " ";
    getOperand(i).print(OS, MAI);
  }
  OS << ">";
}

void MCInst::dump() const {
  print(errs(), 0);
  errs() << "\n";
}
