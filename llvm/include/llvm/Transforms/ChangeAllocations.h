//===- llvm/Transforms/ChangeAllocations.h -----------------------*- C++ -*--=//
//
// This file defines two passes that convert malloc and free instructions to
// calls to and from %malloc & %free function calls.  The LowerAllocations
// transformation is a target dependant tranformation because it depends on the
// size of data types and alignment constraints.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_CHANGEALLOCATIONS_H
#define LLVM_TRANSFORMS_CHANGEALLOCATIONS_H

#include "llvm/Pass.h"
class TargetData;

// LowerAllocations - Turn malloc and free instructions into %malloc and %free
// calls.
//
class LowerAllocations : public BasicBlockPass {
  Method *MallocMeth;   // Methods in the module we are processing
  Method *FreeMeth;     // Initialized by doInitialization

  const TargetData &DataLayout;
public:
  inline LowerAllocations(const TargetData &TD) : DataLayout(TD) {
    MallocMeth = FreeMeth = 0;
  }

  // doPassInitialization - For the lower allocations pass, this ensures that a
  // module contains a declaration for a malloc and a free function.
  //
  bool doInitialization(Module *M);

  // runOnBasicBlock - This method does the actual work of converting
  // instructions over, assuming that the pass has already been initialized.
  //
  bool runOnBasicBlock(BasicBlock *BB);
};

// RaiseAllocations - Turn %malloc and %free calls into the appropriate
// instruction.
//
class RaiseAllocations : public BasicBlockPass {
  Method *MallocMeth;   // Methods in the module we are processing
  Method *FreeMeth;     // Initialized by doPassInitializationVirt
public:
  inline RaiseAllocations() : MallocMeth(0), FreeMeth(0) {}

  // doPassInitialization - For the raise allocations pass, this finds a
  // declaration for malloc and free if they exist.
  //
  bool doInitialization(Module *M);

  // runOnBasicBlock - This method does the actual work of converting
  // instructions over, assuming that the pass has already been initialized.
  //
  bool runOnBasicBlock(BasicBlock *BB);
};

#endif
