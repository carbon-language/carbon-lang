//===- llvm/Transforms/HoistPHIConstants.h - Normalize PHI nodes -*- C++ -*--=//
//
// HoistPHIConstants - Remove literal constants that are arguments of PHI nodes
// by inserting cast instructions in the preceeding basic blocks, and changing
// constant references into references of the casted value.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_HOISTPHICONSTANTS_H
#define LLVM_TRANSFORMS_HOISTPHICONSTANTS_H

class Pass;
Pass *createHoistPHIConstantsPass();

#endif
