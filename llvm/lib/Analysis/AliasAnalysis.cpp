//===- AliasAnalysis.cpp - Generic Alias Analysis Interface Implementation -==//
//
// This file implements the generic AliasAnalysis interface which is used as the
// common interface used by all clients and implementations of alias analysis.
//
// This file also implements the default version of the AliasAnalysis interface
// that is to be used when no other implementation is specified.  This does some
// simple tests that detect obvious cases: two different global pointers cannot
// alias, a global cannot alias a malloc, two different mallocs cannot alias,
// etc.
//
// This alias analysis implementation really isn't very good for anything, but
// it is very fast, and makes a nice clean default implementation.  Because it
// handles lots of little corner cases, other, more complex, alias analysis
// implementations may choose to rely on this pass to resolve these simple and
// easy cases.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/BasicBlock.h"
#include "llvm/iMemory.h"
#include "llvm/Target/TargetData.h"

// Register the AliasAnalysis interface, providing a nice name to refer to.
namespace {
  RegisterAnalysisGroup<AliasAnalysis> Z("Alias Analysis");
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(LoadInst *L, Value *P, unsigned Size) {
  return alias(L->getOperand(0), TD->getTypeSize(L->getType()),
               P, Size) ? Ref : NoModRef;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(StoreInst *S, Value *P, unsigned Size) {
  return alias(S->getOperand(1), TD->getTypeSize(S->getOperand(0)->getType()),
               P, Size) ? Mod : NoModRef;
}


// AliasAnalysis destructor: DO NOT move this to the header file for
// AliasAnalysis or else clients of the AliasAnalysis class may not depend on
// the AliasAnalysis.o file in the current .a file, causing alias analysis
// support to not be included in the tool correctly!
//
AliasAnalysis::~AliasAnalysis() {}

/// setTargetData - Subclasses must call this method to initialize the
/// AliasAnalysis interface before any other methods are called.
///
void AliasAnalysis::InitializeAliasAnalysis(Pass *P) {
  TD = &P->getAnalysis<TargetData>();
}

// getAnalysisUsage - All alias analysis implementations should invoke this
// directly (using AliasAnalysis::getAnalysisUsage(AU)) to make sure that
// TargetData is required by the pass.
void AliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetData>();            // All AA's need TargetData.
}

/// canBasicBlockModify - Return true if it is possible for execution of the
/// specified basic block to modify the value pointed to by Ptr.
///
bool AliasAnalysis::canBasicBlockModify(const BasicBlock &BB,
                                        const Value *Ptr, unsigned Size) {
  return canInstructionRangeModify(BB.front(), BB.back(), Ptr, Size);
}

/// canInstructionRangeModify - Return true if it is possible for the execution
/// of the specified instructions to modify the value pointed to by Ptr.  The
/// instructions to consider are all of the instructions in the range of [I1,I2]
/// INCLUSIVE.  I1 and I2 must be in the same basic block.
///
bool AliasAnalysis::canInstructionRangeModify(const Instruction &I1,
                                              const Instruction &I2,
                                              const Value *Ptr, unsigned Size) {
  assert(I1.getParent() == I2.getParent() &&
         "Instructions not in same basic block!");
  BasicBlock::iterator I = const_cast<Instruction*>(&I1);
  BasicBlock::iterator E = const_cast<Instruction*>(&I2);
  ++E;  // Convert from inclusive to exclusive range.

  for (; I != E; ++I) // Check every instruction in range
    if (getModRefInfo(I, const_cast<Value*>(Ptr), Size) & Mod)
      return true;
  return false;
}

// Because of the way .a files work, we must force the BasicAA implementation to
// be pulled in if the AliasAnalysis classes are pulled in.  Otherwise we run
// the risk of AliasAnalysis being used, but the default implementation not
// being linked into the tool that uses it.
//
extern void BasicAAStub();
static IncludeFile INCLUDE_BASICAA_CPP((void*)&BasicAAStub);

