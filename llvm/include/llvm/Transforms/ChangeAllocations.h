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

class Pass;
class TargetData;

Pass *createLowerAllocationsPass(const TargetData &TD);
Pass *createRaiseAllocationsPass();

#endif
