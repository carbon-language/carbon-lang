//===- AArch64Disassembler.h - Disassembler for AArch64 ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef AArch64DISASSEMBLER_H
#define AArch64DISASSEMBLER_H

#include "llvm/MC/MCDisassembler.h"

namespace llvm {

class MCInst;
class MemoryObject;
class raw_ostream;

class AArch64Disassembler : public MCDisassembler {
public:
  AArch64Disassembler(const MCSubtargetInfo &STI, MCContext &Ctx)
    : MCDisassembler(STI, Ctx) {}

  ~AArch64Disassembler() {}

  /// getInstruction - See MCDisassembler.
  MCDisassembler::DecodeStatus
  getInstruction(MCInst &instr, uint64_t &size, const MemoryObject &region,
                 uint64_t address, raw_ostream &vStream,
                 raw_ostream &cStream) const override;
};

} // namespace llvm

#endif
