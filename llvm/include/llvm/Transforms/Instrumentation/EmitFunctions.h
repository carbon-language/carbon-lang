//===-- EmitFunctions.h - interface to insert instrumentation ----*- C++ -*--=//
//
// Emits function table
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_EMIT_FUNCTIONS_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_EMIT_FUNCTIONS_H
#include "llvm/Pass.h"

//  Create a new pass to add function table
//
Pass *createEmitFunctionTablePass();

#endif
    
