//===-- llvm/MC/MCDisassembler.h - Disassembler interface -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef MCDISASSEMBLER_H
#define MCDISASSEMBLER_H

#include "llvm/System/DataTypes.h"

namespace llvm {
  
class MCInst;
class MemoryObject;
class raw_ostream;
  
struct EDInstInfo;

/// MCDisassembler - Superclass for all disassemblers.  Consumes a memory region
///   and provides an array of assembly instructions.
class MCDisassembler {
public:
  /// Constructor     - Performs initial setup for the disassembler.
  MCDisassembler() {}
  
  virtual ~MCDisassembler();
  
  /// getInstruction  - Returns the disassembly of a single instruction.
  ///
  /// @param instr    - An MCInst to populate with the contents of the 
  ///                   instruction.
  /// @param size     - A value to populate with the size of the instruction, or
  ///                   the number of bytes consumed while attempting to decode
  ///                   an invalid instruction.
  /// @param region   - The memory object to use as a source for machine code.
  /// @param address  - The address, in the memory space of region, of the first
  ///                   byte of the instruction.
  /// @param vStream  - The stream to print warnings and diagnostic messages on.
  /// @return         - True if the instruction is valid; false otherwise.
  virtual bool          getInstruction(MCInst& instr,
                                       uint64_t& size,
                                       const MemoryObject &region,
                                       uint64_t address,
                                       raw_ostream &vStream) const = 0;

  /// getEDInfo - Returns the enhanced insturction information corresponding to
  ///   the disassembler.
  ///
  /// @return         - An array of instruction information, with one entry for
  ///                   each MCInst opcode this disassembler returns.
  ///                   NULL if there is no info for this target.
  virtual EDInstInfo   *getEDInfo() const { return (EDInstInfo*)0; }
};

} // namespace llvm

#endif
