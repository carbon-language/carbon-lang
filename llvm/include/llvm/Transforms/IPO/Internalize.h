//===-- Transforms/IPO/Internalize.h - Mark functions internal ---*- C++ -*--=//
//
// This pass loops over all of the functions in the input module, looking for a
// main function.  If a main function is found, all other functions are marked
// as internal.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORM_IPO_INTERNALIZE_H
#define LLVM_TRANSFORM_IPO_INTERNALIZE_H

class Pass;
Pass *createInternalizePass();

#endif
