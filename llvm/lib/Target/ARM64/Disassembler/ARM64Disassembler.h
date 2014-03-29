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
  ARM64Disassembler(const MCSubtargetInfo &STI) : MCDisassembler(STI) {}

  ~ARM64Disassembler() {}

  /// getInstruction - See MCDisassembler.
  MCDisassembler::DecodeStatus getInstruction(MCInst &instr, uint64_t &size,
                                              const MemoryObject &region,
                                              uint64_t address,
                                              raw_ostream &vStream,
                                              raw_ostream &cStream) const;

  /// tryAddingSymbolicOperand - tryAddingSymbolicOperand trys to add a symbolic
  /// operand in place of the immediate Value in the MCInst.  The immediate
  /// Value has not had any PC adjustment made by the caller. If the instruction
  /// adds the PC to the immediate Value then InstsAddsAddressToValue is true,
  /// else false.  If the getOpInfo() function was set as part of the
  /// setupForSymbolicDisassembly() call then that function is called to get any
  /// symbolic information at the Address for this instrution.  If that returns
  /// non-zero then the symbolic information it returns is used to create an
  /// MCExpr and that is added as an operand to the MCInst.  This function
  /// returns true if it adds an operand to the MCInst and false otherwise.
  bool tryAddingSymbolicOperand(uint64_t Address, int Value,
                                bool InstsAddsAddressToValue, uint64_t InstSize,
                                MCInst &MI, uint32_t insn = 0) const;
};

} // namespace llvm

#endif
