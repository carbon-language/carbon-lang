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

// Forward declaration.
class ARMBasicMCBuilder;

/// Session - Keep track of the IT Block progression.
class Session {
  friend class ARMBasicMCBuilder;
public:
  Session() : ITCounter(0), ITState(0) {}
  ~Session() {}
  /// InitIT - Initializes ITCounter/ITState.
  bool InitIT(unsigned short bits7_0);
  /// UpdateIT - Updates ITCounter/ITState as IT Block progresses.
  void UpdateIT();

private:
  unsigned ITCounter; // Possible values: 0, 1, 2, 3, 4.
  unsigned ITState;   // A2.5.2 Consists of IT[7:5] and IT[4:0] initially.
};

/// ThumbDisassembler - Thumb disassembler for all ARM platforms.
class ThumbDisassembler : public MCDisassembler {
public:
  /// Constructor     - Initializes the disassembler.
  ///
  ThumbDisassembler() :
    MCDisassembler(), SO() {
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
  Session SO;
};

} // namespace llvm
  
#endif
