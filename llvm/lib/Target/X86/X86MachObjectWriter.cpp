//===-- X86MachObjectWriter.cpp - X86 Mach-O Writer -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "llvm/MC/MCMachObjectWriter.h"
using namespace llvm;

namespace {
class X86MachObjectWriter : public MCMachObjectTargetWriter {
public:
  X86MachObjectWriter(bool Is64Bit, uint32_t CPUType,
                      uint32_t CPUSubtype)
    : MCMachObjectTargetWriter(Is64Bit, CPUType, CPUSubtype,
                               /*UseAggressiveSymbolFolding=*/Is64Bit) {}
};
}

MCObjectWriter *llvm::createX86MachObjectWriter(raw_ostream &OS,
                                                bool Is64Bit,
                                                uint32_t CPUType,
                                                uint32_t CPUSubtype) {
  return createMachObjectWriter(new X86MachObjectWriter(Is64Bit,
                                                        CPUType,
                                                        CPUSubtype),
                                OS, /*IsLittleEndian=*/true);
}
