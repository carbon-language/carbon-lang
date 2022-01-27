//===- AArch64Disassembler.h - Disassembler for AArch64 ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_DISASSEMBLER_AARCH64DISASSEMBLER_H
#define LLVM_LIB_TARGET_AARCH64_DISASSEMBLER_AARCH64DISASSEMBLER_H

#include "llvm/MC/MCDisassembler/MCDisassembler.h"

namespace llvm {

class AArch64Disassembler : public MCDisassembler {
public:
  AArch64Disassembler(const MCSubtargetInfo &STI, MCContext &Ctx)
    : MCDisassembler(STI, Ctx) {}

  ~AArch64Disassembler() override = default;

  MCDisassembler::DecodeStatus
  getInstruction(MCInst &Instr, uint64_t &Size, ArrayRef<uint8_t> Bytes,
                 uint64_t Address, raw_ostream &CStream) const override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AARCH64_DISASSEMBLER_AARCH64DISASSEMBLER_H
