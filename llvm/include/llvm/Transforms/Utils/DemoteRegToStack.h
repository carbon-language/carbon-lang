//===- DemoteRegToStack.h - Move a virtual reg. to stack --------*- C++ -*-===//
//
// This file provides the function:
//     AllocaInst* DemoteRegToStack(Instruction& X):
//
// This function takes a virtual register computed by an
// Instruction& X and replaces it with a slot in the stack frame,
// allocated via alloca.  It has to:
// (1) Identify all Phi operations that have X as an operand and
//     transitively other Phis that use such Phis; 
// (2) Store all values merged with X via Phi operations to the stack slot;
// (3) Load the value from the stack slot just before any use of X or any
//     of the Phis that were eliminated; and
// (4) Delete X and all the Phis, which should all now be dead.
//
// Returns the pointer to the alloca inserted to create a stack slot for X.
//
//===----------------------------------------------------------------------===//

class Instruction;
class AllocaInst;

AllocaInst *DemoteRegToStack(Instruction &X);
