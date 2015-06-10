//===-- ARMAsmBackendDarwin.h   ARM Asm Backend Darwin ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMASMBACKENDDARWIN_H
#define LLVM_LIB_TARGET_ARM_ARMASMBACKENDDARWIN_H

#include "llvm/Support/MachO.h"

using namespace llvm;

namespace {
class ARMAsmBackendDarwin : public ARMAsmBackend {
public:
  const MachO::CPUSubTypeARM Subtype;
  ARMAsmBackendDarwin(const Target &T, const Triple &TT,
                      MachO::CPUSubTypeARM st)
      : ARMAsmBackend(T, TT, /* IsLittleEndian */ true), Subtype(st) {
    HasDataInCodeSupport = true;
  }

  MCObjectWriter *createObjectWriter(raw_pwrite_stream &OS) const override {
    return createARMMachObjectWriter(OS, /*Is64Bit=*/false, MachO::CPU_TYPE_ARM,
                                     Subtype);
  }
};
}

#endif
