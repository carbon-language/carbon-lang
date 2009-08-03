//===- lib/Support/Dump.h - Virtual function homes --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the PrefixPrinter virtual function homes.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Dump.h"

using namespace llvm;

PrefixPrinter::~PrefixPrinter() {}
