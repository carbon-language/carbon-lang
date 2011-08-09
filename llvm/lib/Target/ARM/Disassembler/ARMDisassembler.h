//===- ARMDisassembler.h - Disassembler for ARM/Thumb ISA -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the ARM Disassembler.
// It contains the header for ARMDisassembler and ThumbDisassembler, both are
// subclasses of MCDisassembler.
//
//===----------------------------------------------------------------------===//

#ifndef ARMDISASSEMBLER_H
#define ARMDISASSEMBLER_H

#include "llvm/MC/MCDisassembler.h"
#include <vector>

namespace llvm {
  
class MCInst;
class MemoryObject;
class raw_ostream;

struct EDInstInfo;

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

  /// getEDInfo - See MCDisassembler.
  EDInstInfo *getEDInfo() const;
private:
};

/// ARMDisassembler - ARM disassembler for all ARM platforms.
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

  /// getEDInfo - See MCDisassembler.
  EDInstInfo *getEDInfo() const;
private:
  mutable std::vector<unsigned> ITBlock;
  void AddThumbPredicate(MCInst&) const;
  void UpdateThumbVFPPredicate(MCInst&) const;
};


} // namespace llvm

#endif
