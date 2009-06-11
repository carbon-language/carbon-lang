//===-- X86ELFWriterInfo.cpp - ELF Writer Info for the X86 backend --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF writer information for the X86 backend.
//
//===----------------------------------------------------------------------===//

#include "X86ELFWriterInfo.h"
#include "llvm/Function.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

X86ELFWriterInfo::X86ELFWriterInfo(TargetMachine &TM)
  : TargetELFWriterInfo(TM) {
    bool is64Bit = TM.getTargetData()->getPointerSizeInBits() == 64;
    EMachine = is64Bit ? EM_X86_64 : EM_386;
  }

X86ELFWriterInfo::~X86ELFWriterInfo() {}

unsigned X86ELFWriterInfo::getFunctionAlignment(const Function *F) const {
  unsigned FnAlign = 4;

  if (F->hasFnAttr(Attribute::OptimizeForSize))
    FnAlign = 1;

  if (F->getAlignment())
    FnAlign = Log2_32(F->getAlignment());

  return (1 << FnAlign);
}
