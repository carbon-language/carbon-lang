//===-- SparcV9TmpInstr.h ---------------------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Definition of class for temporary intermediate values used within the current
// SparcV9 backend.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCV9TMPINSTR_H
#define SPARCV9TMPINSTR_H

#include "llvm/Instruction.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"

namespace llvm {

/// TmpInstruction - This class represents temporary intermediate
/// values used within the SparcV9 machine code for an LLVM instruction.
/// 
class TmpInstruction : public Instruction {
  TmpInstruction(const TmpInstruction &TI)
    : Instruction(TI.getType(), TI.getOpcode()) {
    if (!TI.Operands.empty()) {
      Operands.push_back(Use(TI.Operands[0], this));
      if (TI.Operands.size() == 2)
        Operands.push_back(Use(TI.Operands[1], this));
      else
        assert(0 && "Bad # operands to TmpInstruction!");
    }
  }
public:
  // Constructor that uses the type of S1 as the type of the temporary.
  // s1 must be a valid value.  s2 may be NULL.
  TmpInstruction(MachineCodeForInstruction &mcfi,
                 Value *s1, Value *s2 = 0, const std::string &name = "");
                 
  // Constructor that uses the type of S1 as the type of the temporary,
  // but does not require a MachineCodeForInstruction.
  // s1 must be a valid value.  s2 may be NULL.
  TmpInstruction(Value *s1, Value *s2 = 0, const std::string &name = "");

  // Constructor that requires the type of the temporary to be specified.
  // Both S1 and S2 may be NULL.
  TmpInstruction(MachineCodeForInstruction& mcfi,
                 const Type *Ty, Value *s1 = 0, Value* s2 = 0,
                 const std::string &name = "");
  
  virtual Instruction *clone() const {
    assert(0 && "Cannot clone TmpInstructions!");
    return 0;
  }
  virtual const char *getOpcodeName() const { return "TmpInstruction"; }
  
  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const TmpInstruction *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::UserOp1);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

} // End llvm namespace

#endif
