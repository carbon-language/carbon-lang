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
class Module;
class Function;
class BasicBlock;
class Value;
class CallInst;
class ReturnInst;

// Clone OldFunc into NewFunc, transforming the old arguments into references to
// ArgMap values.  Note that if NewFunc already has basic blocks, the ones
// cloned into it will be added to the end of the function.  This function fills
// in a list of return instructions, and can optionally append the specified
// suffix to all values cloned.
//
void CloneFunctionInto(Function *NewFunc, const Function *OldFunc,
                       const std::vector<Value*> &ArgMap,
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
