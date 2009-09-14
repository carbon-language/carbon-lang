//===-- MCInstPrinter.cpp - Convert an MCInst to target assembly syntax ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInstPrinter.h"
using namespace llvm;

MCInstPrinter::~MCInstPrinter() {
}