//===-- X86AsmBackend.cpp - X86 Assembler Backend -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetAsmBackend.h"
#include "X86.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmBackend.h"
using namespace llvm;

namespace {

class X86AsmBackend : public TargetAsmBackend {
public:
  X86AsmBackend(const Target &T, MCAssembler &A)
    : TargetAsmBackend(T) {}
};

}

TargetAsmBackend *llvm::createX86_32AsmBackend(const Target &T,
                                               MCAssembler &A) {
  return new X86AsmBackend(T, A);
}

TargetAsmBackend *llvm::createX86_64AsmBackend(const Target &T,
                                               MCAssembler &A) {
  return new X86AsmBackend(T, A);
}
