//===-- MCInstPrinter.cpp - Convert an MCInst to target assembly syntax ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

MCInstPrinter::~MCInstPrinter() {
}

/// getOpcodeName - Return the name of the specified opcode enum (e.g.
/// "MOV32ri") or empty if we can't resolve it.
StringRef MCInstPrinter::getOpcodeName(unsigned Opcode) const {
  return "";
}

void MCInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  assert(0 && "Target should implement this");
}

void MCInstPrinter::printAnnotations(const MCInst *MI, raw_ostream &OS) {
  for (unsigned i = 0, e = MI->getNumAnnotations(); i != e; ++i) {
    OS << MI->getAnnotation(i) << "\n";
  }
}
