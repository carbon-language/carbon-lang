//===-- TargetAsmParser.cpp - Target Assembly Parser -----------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetAsmParser.h"
using namespace llvm;

TargetAsmParser::TargetAsmParser(const Target &T) 
  : TheTarget(T), AvailableFeatures(0)
{
}

TargetAsmParser::~TargetAsmParser() {
}
