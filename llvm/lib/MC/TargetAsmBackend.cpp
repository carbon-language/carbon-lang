//===-- TargetAsmBackend.cpp - Target Assembly Backend ---------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetAsmBackend.h"
using namespace llvm;

TargetAsmBackend::TargetAsmBackend(const Target &T)
  : TheTarget(T)
{
}

TargetAsmBackend::~TargetAsmBackend() {
}
