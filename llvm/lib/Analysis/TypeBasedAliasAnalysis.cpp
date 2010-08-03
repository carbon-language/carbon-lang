//===- TypeBasedAliasAnalysis.cpp - Type-Based Alias Analysis -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TypeBasedAliasAnalysis pass, which implements
// metadata-based TBAA.
//
// In LLVM IR, memory does not have types, so LLVM's own type system is not
// suitable for doing TBAA. Instead, metadata is added to the IR to describe
// a type system of a higher level language.
//
// This pass is language-independent. The type system is encoded in
// metadata. This allows this pass to support typical C and C++ TBAA, but
// it can also support custom aliasing behavior for other languages.
//
// This is a work-in-progress. It doesn't work yet, and the metadata
// format isn't stable.
//
// TODO: getModRefBehavior. The AliasAnalysis infrastructure will need to
//       be extended.
// TODO: AA chaining
// TODO: struct fields
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Module.h"
#include "llvm/Metadata.h"
#include "llvm/Pass.h"
using namespace llvm;

namespace {
  /// TBAANode - This is a simple wrapper around an MDNode which provides a
  /// higher-level interface by hiding the details of how alias analysis
  /// information is encoded in its operands.
  class TBAANode {
    const MDNode *Node;

  public:
    TBAANode() : Node(0) {}
    explicit TBAANode(MDNode *N) : Node(N) {}

    /// getNode - Get the MDNode for this TBAANode.
    const MDNode *getNode() const { return Node; }

    /// getParent - Get this TBAANode's Alias DAG parent.
    TBAANode getParent() const {
      if (Node->getNumOperands() < 2)
        return TBAANode();
      MDNode *P = dyn_cast<MDNode>(Node->getOperand(1));
      if (!P)
        return TBAANode();
      // Ok, this node has a valid parent. Return it.
      return TBAANode(P);
    }

    /// TypeIsImmutable - Test if this TBAANode represents a type for objects
    /// which are not modified (by any means) in the context where this
    /// AliasAnalysis is relevant.
    bool TypeIsImmutable() const {
      if (Node->getNumOperands() < 3)
        return false;
      ConstantInt *CI = dyn_cast<ConstantInt>(Node->getOperand(2));
      if (!CI)
        return false;
      // TODO: Think about the encoding.
      return CI->isOne();
    }
  };
}

namespace {
  /// TypeBasedAliasAnalysis - This is a simple alias analysis
  /// implementation that uses TypeBased to answer queries.
  class TypeBasedAliasAnalysis : public ImmutablePass,
                                 public AliasAnalysis {
  public:
    static char ID; // Class identification, replacement for typeinfo
    TypeBasedAliasAnalysis() : ImmutablePass(&ID) {}

    /// getAdjustedAnalysisPointer - This method is used when a pass implements
    /// an analysis interface through multiple inheritance.  If needed, it
    /// should override this to adjust the this pointer as needed for the
    /// specified pass info.
    virtual void *getAdjustedAnalysisPointer(const PassInfo *PI) {
      if (PI->isPassID(&AliasAnalysis::ID))
        return (AliasAnalysis*)this;
      return this;
    }

  private:
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual AliasResult alias(const Value *V1, unsigned V1Size,
                              const Value *V2, unsigned V2Size);
    virtual bool pointsToConstantMemory(const Value *P);
  };
}  // End of anonymous namespace

// Register this pass...
char TypeBasedAliasAnalysis::ID = 0;
INITIALIZE_AG_PASS(TypeBasedAliasAnalysis, AliasAnalysis, "tbaa",
                   "Type-Based Alias Analysis", false, true, false);

ImmutablePass *llvm::createTypeBasedAliasAnalysisPass() {
  return new TypeBasedAliasAnalysis();
}

void
TypeBasedAliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AliasAnalysis::getAnalysisUsage(AU);
}

AliasAnalysis::AliasResult
TypeBasedAliasAnalysis::alias(const Value *A, unsigned ASize,
                              const Value *B, unsigned BSize) {
  // Currently, metadata can only be attached to Instructions.
  const Instruction *AI = dyn_cast<Instruction>(A);
  if (!AI) return MayAlias;
  const Instruction *BI = dyn_cast<Instruction>(B);
  if (!BI) return MayAlias;

  // Get the attached MDNodes. If either value lacks a tbaa MDNode, we must
  // be conservative.
  MDNode *AM =
    AI->getMetadata(AI->getParent()->getParent()->getParent()
                      ->getMDKindID("tbaa"));
  if (!AM) return MayAlias;
  MDNode *BM =
    BI->getMetadata(BI->getParent()->getParent()->getParent()
                      ->getMDKindID("tbaa"));
  if (!BM) return MayAlias;

  // Keep track of the root node for A and B.
  TBAANode RootA, RootB;

  // Climb the DAG from A to see if we reach B.
  for (TBAANode T(AM); ; ) {
    if (T.getNode() == BM)
      // B is an ancestor of A.
      return MayAlias;

    RootA = T;
    T = T.getParent();
    if (!T.getNode())
      break;
  }

  // Climb the DAG from B to see if we reach A.
  for (TBAANode T(BM); ; ) {
    if (T.getNode() == AM)
      // A is an ancestor of B.
      return MayAlias;

    RootB = T;
    T = T.getParent();
    if (!T.getNode())
      break;
  }

  // Neither node is an ancestor of the other.
  
  // If they have the same root, then we've proved there's no alias.
  if (RootA.getNode() == RootB.getNode())
    return NoAlias;

  // If they have different roots, they're part of different potentially
  // unrelated type systems, so we must be conservative.
  return MayAlias;
}

bool TypeBasedAliasAnalysis::pointsToConstantMemory(const Value *P) {
  // Currently, metadata can only be attached to Instructions.
  const Instruction *I = dyn_cast<Instruction>(P);
  if (!I) return false;

  MDNode *M =
    I->getMetadata(I->getParent()->getParent()->getParent()
                    ->getMDKindID("tbaa"));
  if (!M) return false;

  // If this is an "immutable" type, we can assume the pointer is pointing
  // to constant memory.
  return TBAANode(M).TypeIsImmutable();
}
