//===- Cloning.h - Clone various parts of LLVM programs ---------*- C++ -*-===//
//
// This file defines various functions that are used to clone chunks of LLVM
// code for various purposes.  This varies from copying whole modules into new
// modules, to cloning functions with different arguments, to inlining
// functions, to copying basic blocks to support loop unrolling or superblock
// formation, etc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_CLONING_H
#define LLVM_TRANSFORMS_UTILS_CLONING_H

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

/// CloneBasicBlock - Return a copy of the specified basic block, but without
/// embedding the block into a particular function.  The block returned is an
/// exact copy of the specified basic block, without any remapping having been
/// performed.  Because of this, this is only suitable for applications where
/// the basic block will be inserted into the same function that it was cloned
/// from (loop unrolling would use this, for example).
///
/// Also, note that this function makes a direct copy of the basic block, and
/// can thus produce illegal LLVM code.  In particular, it will copy any PHI
/// nodes from the original block, even though there are no predecessors for the
/// newly cloned block (thus, phi nodes will have to be updated).  Also, this
/// block will branch to the old successors of the original block: these
/// successors will have to have any PHI nodes updated to account for the new
/// incoming edges.
///
/// The correlation between instructions in the source and result basic blocks
/// is recorded in the ValueMap map.
///
/// If you have a particular suffix you'd like to use to add to any cloned
/// names, specify it as the optional second parameter.
///
BasicBlock *CloneBasicBlock(const BasicBlock *BB,
                            std::map<const Value*, Value*> &ValueMap,
                            const char *NameSuffix = "");


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

/// Clone OldFunc into NewFunc, transforming the old arguments into references
/// to ArgMap values.  Note that if NewFunc already has basic blocks, the ones
/// cloned into it will be added to the end of the function.  This function
/// fills in a list of return instructions, and can optionally append the
/// specified suffix to all values cloned.
///
void CloneFunctionInto(Function *NewFunc, const Function *OldFunc,
                       std::map<const Value*, Value*> &ValueMap,
                       std::vector<ReturnInst*> &Returns,
                       const char *NameSuffix = "");


/// InlineFunction - This function inlines the called function into the basic
/// block of the caller.  This returns true if it is not possible to inline this
/// call.  The program is still in a well defined state if this occurs though.
///
/// Note that this only does one level of inlining.  For example, if the 
/// instruction 'call B' is inlined, and 'B' calls 'C', then the call to 'C' now
/// exists in the instruction stream.  Similiarly this will inline a recursive
/// function by one level.
///
bool InlineFunction(CallInst *C);


/// CloneTrace - Returns a copy of the specified trace. It removes internal phi
/// nodes, copies the basic blocks, remaps variables, and returns a new vector
/// of basic blocks (the cloned trace).
///
std::vector<BasicBlock *> cloneTrace(std::vector<BasicBlock*> &origTrace);

#endif
