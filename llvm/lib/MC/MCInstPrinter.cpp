//===-- MCInstPrinter.cpp - Convert an MCInst to target assembly syntax ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInstPrinter.h"
#include "llvm/ADT/StringRef.h"
using namespace llvm;

MCInstPrinter::~MCInstPrinter() {
}

/// getOpcodeName - Return the name of the specified opcode enum (e.g.
/// "MOV32ri") or empty if we can't resolve it.
StringRef MCInstPrinter::getOpcodeName(unsigned Opcode) const {
  return "";
}
