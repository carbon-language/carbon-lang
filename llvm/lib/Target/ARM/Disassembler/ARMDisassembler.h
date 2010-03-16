//===- X86Disassembler.h - Disassembler for x86 and x86_64 ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef ARMDISASSEMBLER_H
#define ARMDISASSEMBLER_H

#include "llvm/MC/MCDisassembler.h"

namespace llvm {
  
class MCInst;
class MemoryObject;
class raw_ostream;
  
namespace ARMDisassembler {

/// ARMDisassembler - ARM disassembler for all ARM platforms.
class ARMDisassembler : public MCDisassembler {
public:
  /// Constructor     - Initializes the disassembler.
  ///
  ARMDisassembler() :
    MCDisassembler() {
  }

  ~ARMDisassembler() {
  }

  /// getInstruction - See MCDisassembler.
  bool getInstruction(MCInst &instr,
                      uint64_t &size,
                      const MemoryObject &region,
                      uint64_t address,
                      raw_ostream &vStream) const;
private:
};

/// ThumbDisassembler - Thumb disassembler for all ARM platforms.
class ThumbDisassembler : public MCDisassembler {
public:
  /// Constructor     - Initializes the disassembler.
  ///
  ThumbDisassembler() :
    MCDisassembler() {
  }

  ~ThumbDisassembler() {
  }

  /// getInstruction - See MCDisassembler.
  bool getInstruction(MCInst &instr,
                      uint64_t &size,
                      const MemoryObject &region,
                      uint64_t address,
                      raw_ostream &vStream) const;
private:
};

} // namespace ARMDisassembler
  
} // namespace llvm
  
#endif
