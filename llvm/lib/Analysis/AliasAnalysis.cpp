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
#include "llvm/iOther.h"
#include "llvm/Constants.h"
#include "llvm/GlobalValue.h"
#include "llvm/DerivedTypes.h"
#include "Support/Statistic.h"

// Register the AliasAnalysis interface, providing a nice name to refer to.
namespace {
  RegisterAnalysisGroup<AliasAnalysis> Z("Alias Analysis");
  Statistic<> NumNoAlias  ("basic-aa", "Number of 'no alias' replies");
  Statistic<> NumMayAlias ("basic-aa", "Number of 'may alias' replies");
  Statistic<> NumMustAlias("basic-aa", "Number of 'must alias' replies");
}

// CanModify - Define a little visitor class that is used to check to see if
// arbitrary chunks of code can modify a specified pointer.
//
namespace {
  struct CanModify : public InstVisitor<CanModify, bool> {
    AliasAnalysis &AA;
    const Value *Ptr;

    CanModify(AliasAnalysis *aa, const Value *ptr)
      : AA(*aa), Ptr(ptr) {}

    bool visitInvokeInst(InvokeInst &II) {
      return AA.canInvokeModify(II, Ptr);
    }
    bool visitCallInst(CallInst &CI) {
      return AA.canCallModify(CI, Ptr);
    }
    bool visitStoreInst(StoreInst &SI) {
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

/// canBasicBlockModify - Return true if it is possible for execution of the
/// specified basic block to modify the value pointed to by Ptr.
///
bool AliasAnalysis::canBasicBlockModify(const BasicBlock &bb,
                                        const Value *Ptr) {
  CanModify CM(this, Ptr);
  BasicBlock &BB = const_cast<BasicBlock&>(bb);

  for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I)
    if (CM.visit(I))        // Check every instruction in the basic block...
      return true;

  return false;
}

/// canInstructionRangeModify - Return true if it is possible for the execution
/// of the specified instructions to modify the value pointed to by Ptr.  The
/// instructions to consider are all of the instructions in the range of [I1,I2]
/// INCLUSIVE.  I1 and I2 must be in the same basic block.
///
bool AliasAnalysis::canInstructionRangeModify(const Instruction &I1,
                                              const Instruction &I2,
                                              const Value *Ptr) {
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

static const Value *getUnderlyingObject(const Value *V) {
  if (!isa<PointerType>(V->getType())) return 0;

  // If we are at some type of object... return it.
  if (hasUniqueAddress(V)) return V;
  
  // Traverse through different addressing mechanisms...
  if (const Instruction *I = dyn_cast<Instruction>(V)) {
    if (isa<CastInst>(I) || isa<GetElementPtrInst>(I))
      return getUnderlyingObject(I->getOperand(0));
  }
  return 0;
}

static inline AliasAnalysis::Result MustAlias() {
  ++NumMustAlias;
  return AliasAnalysis::MustAlias;
}

static inline AliasAnalysis::Result MayAlias() {
  ++NumMayAlias;
  return AliasAnalysis::MayAlias;
}

static inline AliasAnalysis::Result NoAlias() {
  ++NumNoAlias;
  return AliasAnalysis::NoAlias;
}

// alias - Provide a bunch of ad-hoc rules to disambiguate in common cases, such
// as array references.  Note that this function is heavily tail recursive.
// Hopefully we have a smart C++ compiler.  :)
//
AliasAnalysis::Result BasicAliasAnalysis::alias(const Value *V1,
                                                const Value *V2) {
  // Strip off constant pointer refs if they exist
  if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(V1))
    V1 = CPR->getValue();
  if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(V2))
    V2 = CPR->getValue();

  // Are we checking for alias of the same value?
  if (V1 == V2) return ::MustAlias();

  if ((!isa<PointerType>(V1->getType()) || !isa<PointerType>(V2->getType())) &&
      V1->getType() != Type::LongTy && V2->getType() != Type::LongTy)
    return ::NoAlias();  // Scalars cannot alias each other

  // Strip off cast instructions...
  if (const Instruction *I = dyn_cast<CastInst>(V1))
    return alias(I->getOperand(0), V2);
  if (const Instruction *I = dyn_cast<CastInst>(V2))
    return alias(I->getOperand(0), V1);

  // If we have two gep instructions with identical indices, return an alias
  // result equal to the alias result of the original pointer...
  //
  if (const GetElementPtrInst *GEP1 = dyn_cast<GetElementPtrInst>(V1))
    if (const GetElementPtrInst *GEP2 = dyn_cast<GetElementPtrInst>(V2))
      if (GEP1->getNumOperands() == GEP2->getNumOperands() &&
          GEP1->getOperand(0)->getType() == GEP2->getOperand(0)->getType()) {
        if (std::equal(GEP1->op_begin()+1, GEP1->op_end(), GEP2->op_begin()+1))
          return alias(GEP1->getOperand(0), GEP2->getOperand(0));

        // If all of the indexes to the getelementptr are constant, but
        // different (well we already know they are different), then we know
        // that there cannot be an alias here if the two base pointers DO alias.
        //
        bool AllConstant = true;
        for (unsigned i = 1, e = GEP1->getNumOperands(); i != e; ++i)
          if (!isa<Constant>(GEP1->getOperand(i)) ||
              !isa<Constant>(GEP2->getOperand(i))) {
            AllConstant = false;
            break;
          }

        // If we are all constant, then look at where the the base pointers
        // alias.  If they are known not to alias, then we are dealing with two
        // different arrays or something, so no alias is possible.  If they are
        // known to be the same object, then we cannot alias because we are
        // indexing into a different part of the object.  As usual, MayAlias
        // doesn't tell us anything.
        //
        if (AllConstant &&
            alias(GEP1->getOperand(0), GEP2->getOperand(1)) != MayAlias)
            return ::NoAlias();
      }

  // Figure out what objects these things are pointing to if we can...
  const Value *O1 = getUnderlyingObject(V1);
  const Value *O2 = getUnderlyingObject(V2);

  // Pointing at a discernable object?
  if (O1 && O2) {
    // If they are two different objects, we know that we have no alias...
    if (O1 != O2) return ::NoAlias();

    // If they are the same object, they we can look at the indexes.  If they
    // index off of the object is the same for both pointers, they must alias.
    // If they are provably different, they must not alias.  Otherwise, we can't
    // tell anything.
  } else if (O1 && isa<ConstantPointerNull>(V2)) {
    return ::NoAlias();                    // Unique values don't alias null
  } else if (O2 && isa<ConstantPointerNull>(V1)) {
    return ::NoAlias();                    // Unique values don't alias null
  }

  return MayAlias;
}
