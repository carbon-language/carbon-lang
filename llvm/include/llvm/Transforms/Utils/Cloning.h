//===- Cloning.h - Clone various parts of LLVM programs ---------*- C++ -*-===//
//
// This file defines various functions that are used to clone chunks of LLVM
// code for various purposes.  This varies from copying whole modules into new
// modules, to cloning functions with different arguments, to inlining
// functions, to copying basic blocks to support loop unrolling or superblock
// formation, etc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTIlS_CLONING_H
#define LLVM_TRANSFORMS_UTIlS_CLONING_H

#include <vector>
#include <map>
class Module;
class Function;
class BasicBlock;
class Value;
class CallInst;
class ReturnInst;

/// CloneModule - Return an exact copy of the specified module
///
Module *CloneModule(const Module *M);

/// CloneFunction - Return a copy of the specified function, but without
/// embedding the function into another module.  Also, any references specified
/// in the ValueMap are changed to refer to their mapped value instead of the
/// original one.  If any of the arguments to the function are in the ValueMap,
/// the arguments are deleted from the resultant function.  The ValueMap is
/// updated to include mappings from all of the instructions and basicblocks in
/// the function from their old to new values.
///
Function *CloneFunction(const Function *F,
                        std::map<const Value*, Value*> &ValueMap);

/// CloneFunction - Version of the function that doesn't need the ValueMap.
///
inline Function *CloneFunction(const Function *F) {
  std::map<const Value*, Value*> ValueMap;
  return CloneFunction(F, ValueMap);
}

// Clone OldFunc into NewFunc, transforming the old arguments into references to
// ArgMap values.  Note that if NewFunc already has basic blocks, the ones
// cloned into it will be added to the end of the function.  This function fills
// in a list of return instructions, and can optionally append the specified
// suffix to all values cloned.
//
void CloneFunctionInto(Function *NewFunc, const Function *OldFunc,
                       std::map<const Value*, Value*> &ValueMap,
                       std::vector<ReturnInst*> &Returns,
                       const char *NameSuffix = "");


// InlineFunction - This function forcibly inlines the called function into the
// basic block of the caller.  This returns true if it is not possible to inline
// this call.  The program is still in a well defined state if this occurs 
// though.
//
// Note that this only does one level of inlining.  For example, if the 
// instruction 'call B' is inlined, and 'B' calls 'C', then the call to 'C' now 
// exists in the instruction stream.  Similiarly this will inline a recursive
// function by one level.
//
bool InlineFunction(CallInst *C);

#endif
