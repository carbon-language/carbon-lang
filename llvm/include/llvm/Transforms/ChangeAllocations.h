//===- llvm/Transforms/LowerAllocations.h - Remove Malloc & Free -*- C++ -*--=//
//
// This file defines the interface to a pass that lowers malloc and free
// instructions to calls to %malloc & %free functions.  This transformation is
// a target dependant tranformation because we depend on the size of data types
// and alignment constraints.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_LOWERALLOCATIONS_H
#define LLVM_TRANSFORMS_LOWERALLOCATIONS_H

#include "llvm/Pass.h"
class TargetData;

class LowerAllocations : public BasicBlockPass {
  Method *MallocMeth;   // Methods in the module we are processing
  Method *FreeMeth;     // Initialized by doPassInitializationVirt

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

#endif
