//===- ARM64Disassembler.h - Disassembler for ARM64 -------------*- C++ -*-===//
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

#ifndef ARM64DISASSEMBLER_H
#define ARM64DISASSEMBLER_H

#include "llvm/MC/MCDisassembler.h"

namespace llvm {

class MCInst;
class MemoryObject;
class raw_ostream;

class ARM64Disassembler : public MCDisassembler {
public:
  ARM64Disassembler(const MCSubtargetInfo &STI, MCContext &Ctx)
    : MCDisassembler(STI, Ctx) {}

  ~ARM64Disassembler() {}

  /// getInstruction - See MCDisassembler.
  MCDisassembler::DecodeStatus getInstruction(MCInst &instr, uint64_t &size,
                                              const MemoryObject &region,
                                              uint64_t address,
                                              raw_ostream &vStream,
                                              raw_ostream &cStream) const;
};

} // namespace llvm

#endif
