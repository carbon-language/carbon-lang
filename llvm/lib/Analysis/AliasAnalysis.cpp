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

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/BasicBlock.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/iMemory.h"
#include "llvm/Constants.h"
#include "llvm/GlobalValue.h"
#include "llvm/Pass.h"

// Register the AliasAnalysis interface, providing a nice name to refer to.
static RegisterAnalysisGroup<AliasAnalysis> X("Alias Analysis");

// CanModify - Define a little visitor class that is used to check to see if
// arbitrary chunks of code can modify a specified pointer.
//
namespace {
  struct CanModify : public InstVisitor<CanModify, bool> {
    const AliasAnalysis &AA;
    const Value *Ptr;

    CanModify(const AliasAnalysis *aa, const Value *ptr)
      : AA(*aa), Ptr(ptr) {}

    bool visitInvokeInst(InvokeInst &II) {
      return AA.canInvokeModify(II, Ptr);
    }
    bool visitCallInst(CallInst &CI) {
      return AA.canCallModify(CI, Ptr);
    }
    bool visitStoreInst(StoreInst &SI) {
      assert(!SI.hasIndices() && "Only support stores without indexing!");
      return AA.alias(Ptr, SI.getOperand(1));
    }

    // Other instructions do not alias anything.
    bool visitInstruction(Instruction &I) { return false; }
  };
}

// AliasAnalysis destructor: DO NOT move this to the header file for
// AliasAnalysis or else clients of the AliasAnalysis class may not depend on
// the AliasAnalysis.o file in the current .a file, causing alias analysis
// support to not be included in the tool correctly!
//
AliasAnalysis::~AliasAnalysis() {}

// canBasicBlockModify - Return true if it is possible for execution of the
// specified basic block to modify the value pointed to by Ptr.
//
bool AliasAnalysis::canBasicBlockModify(const BasicBlock &bb,
                                        const Value *Ptr) const {
  CanModify CM(this, Ptr);
  BasicBlock &BB = const_cast<BasicBlock&>(bb);

  for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I)
    if (CM.visit(I))        // Check every instruction in the basic block...
      return true;

  return false;
}

// canInstructionRangeModify - Return true if it is possible for the execution
// of the specified instructions to modify the value pointed to by Ptr.  The
// instructions to consider are all of the instructions in the range of [I1,I2]
// INCLUSIVE.  I1 and I2 must be in the same basic block.
//
bool AliasAnalysis::canInstructionRangeModify(const Instruction &I1,
                                              const Instruction &I2,
                                              const Value *Ptr) const {
  assert(I1.getParent() == I2.getParent() &&
         "Instructions not in same basic block!");
  CanModify CM(this, Ptr);
  BasicBlock::iterator I = const_cast<Instruction*>(&I1);
  BasicBlock::iterator E = const_cast<Instruction*>(&I2);
  ++E;  // Convert from inclusive to exclusive range.

  for (; I != E; ++I)
    if (CM.visit(I))        // Check every instruction in the basic block...
      return true;

  return false;
}

//===----------------------------------------------------------------------===//
// BasicAliasAnalysis Pass Implementation
//===----------------------------------------------------------------------===//
//
// Because of the way .a files work, the implementation of the
// BasicAliasAnalysis class MUST be in the AliasAnalysis file itself, or else we
// run the risk of AliasAnalysis being used, but the default implementation not
// being linked into the tool that uses it.  As such, we register and implement
// the class here.
//
namespace {
  // Register this pass...
  RegisterOpt<BasicAliasAnalysis>
  X("basicaa", "Basic Alias Analysis (default AA impl)");

  // Declare that we implement the AliasAnalysis interface
  RegisterAnalysisGroup<AliasAnalysis, BasicAliasAnalysis, true> Y;
}  // End of anonymous namespace



// hasUniqueAddress - Return true if the 
static inline bool hasUniqueAddress(const Value *V) {
  return isa<GlobalValue>(V) || isa<MallocInst>(V) || isa<AllocaInst>(V);
}

AliasAnalysis::Result BasicAliasAnalysis::alias(const Value *V1,
                                                const Value *V2) const {
  // Strip off constant pointer refs if they exist
  if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(V1))
    V1 = CPR->getValue();
  if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(V2))
    V2 = CPR->getValue();

  // Are we checking for alias of the same value?
  if (V1 == V2) return MustAlias;

  if (!isa<PointerType>(V1->getType()) || !isa<PointerType>(V2->getType()))
    return NoAlias;  // Scalars cannot alias each other

  bool V1Unique = hasUniqueAddress(V1);
  bool V2Unique = hasUniqueAddress(V2);

  if (V1Unique && V2Unique)
    return NoAlias;         // Can't alias if they are different unique values

  if ((V1Unique && isa<ConstantPointerNull>(V2)) ||
      (V2Unique && isa<ConstantPointerNull>(V1)))
    return NoAlias;         // Unique values don't alias null

  // TODO: Handle getelementptr with nonzero offset

  return MayAlias;
}
