//===-- MInstruction.cpp - Implementation code for the MInstruction class -===//
//
// This file contains a printer that converts from our internal representation
// of LLVM code to a nice human readable form that is suitable for debuggging.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MBasicBlock.h"

/// MInstruction ctor - Create a new instruction, and append it to the
/// specified basic block.
///
MInstruction::MInstruction(MBasicBlock *BB, unsigned O, unsigned D)
  : Opcode(O), Dest(D) {
  // Add this instruction to the specified basic block
  BB->getInstList().push_back(this);
}


/// addOperand - Add a new operand to the instruction with the specified value
/// and interpretation.
///
void MInstruction::addOperand(unsigned Value, MOperand::Interpretation Ty) {
  if (Operands.size() < 4) {
    OperandInterpretation[Operands.size()] = Ty;  // Save interpretation
  } else {
    assert(Ty == MOperand::Register &&
           "Trying to add 5th operand that is not a register to MInstruction!");
  }
  Operands.push_back(Value);
}
