//===-- ARMAsmBackendWinCOFF.h - ARM Asm Backend WinCOFF --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMASMBACKENDWINCOFF_H
#define LLVM_LIB_TARGET_ARM_ARMASMBACKENDWINCOFF_H

using namespace llvm;

namespace {
class ARMAsmBackendWinCOFF : public ARMAsmBackend {
public:
  ARMAsmBackendWinCOFF(const Target &T, StringRef Triple)
      : ARMAsmBackend(T, Triple, true) {}
  MCObjectWriter *createObjectWriter(raw_ostream &OS) const override {
    return createARMWinCOFFObjectWriter(OS, /*Is64Bit=*/false);
  }
};
}

#endif
