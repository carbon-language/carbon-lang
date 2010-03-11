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
  X86AsmBackend(const Target &T)
    : TargetAsmBackend(T) {}
};

class DarwinX86AsmBackend : public X86AsmBackend {
public:
  DarwinX86AsmBackend(const Target &T)
    : X86AsmBackend(T) {}

  virtual bool hasAbsolutizedSet() const { return true; }

  virtual bool hasScatteredSymbols() const { return true; }
};

}

TargetAsmBackend *llvm::createX86_32AsmBackend(const Target &T,
                                               const std::string &TT) {
  switch (Triple(TT).getOS()) {
  case Triple::Darwin:
    return new DarwinX86AsmBackend(T);
  default:
    return new X86AsmBackend(T);
  }
}

TargetAsmBackend *llvm::createX86_64AsmBackend(const Target &T,
                                               const std::string &TT) {
  switch (Triple(TT).getOS()) {
  case Triple::Darwin:
    return new DarwinX86AsmBackend(T);
  default:
    return new X86AsmBackend(T);
  }
}
