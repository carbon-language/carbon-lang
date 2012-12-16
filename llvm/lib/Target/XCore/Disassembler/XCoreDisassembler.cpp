//===- XCoreDisassembler.cpp - Disassembler for XCore -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the XCore Disassembler.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

typedef MCDisassembler::DecodeStatus DecodeStatus;

namespace {

/// XCoreDisassembler - a disasembler class for XCore.
class XCoreDisassembler : public MCDisassembler {
public:
  /// Constructor     - Initializes the disassembler.
  ///
  XCoreDisassembler(const MCSubtargetInfo &STI) :
    MCDisassembler(STI) {}

  /// getInstruction - See MCDisassembler.
  virtual DecodeStatus getInstruction(MCInst &instr,
                                      uint64_t &size,
                                      const MemoryObject &region,
                                      uint64_t address,
                                      raw_ostream &vStream,
                                      raw_ostream &cStream) const;
};

}

MCDisassembler::DecodeStatus
XCoreDisassembler::getInstruction(MCInst &instr,
                                  uint64_t &Size,
                                  const MemoryObject &Region,
                                  uint64_t Address,
                                  raw_ostream &vStream,
                                  raw_ostream &cStream) const {
  return Fail;
}

namespace llvm {
  extern Target TheXCoreTarget;
}

static MCDisassembler *createXCoreDisassembler(const Target &T,
                                               const MCSubtargetInfo &STI) {
  return new XCoreDisassembler(STI);
}

extern "C" void LLVMInitializeXCoreDisassembler() {
  // Register the disassembler.
  TargetRegistry::RegisterMCDisassembler(TheXCoreTarget,
                                         createXCoreDisassembler);
}
