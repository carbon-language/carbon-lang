//===- X86InstructionInfo.h - X86 Instruction Information ---------*-C++-*-===//
//
// This file contains the X86 implementation of the MInstructionInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86INSTRUCTIONINFO_H
#define X86INSTRUCTIONINFO_H

#include "llvm/Target/InstructionInfo.h"
#include "X86RegisterInfo.h"

class X86InstructionInfo : public MInstructionInfo {
  const X86RegisterInfo RI;
public:
  X86InstructionInfo();

  /// getRegisterInfo - MInstructionInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MRegisterInfo &getRegisterInfo() const { return RI; }

  /// print - Print out an x86 instruction in GAS syntax
  ///
  virtual void print(const MInstruction *MI, std::ostream &O) const;
};


#endif
