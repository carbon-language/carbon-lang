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
// The current metadata format is very simple. MDNodes have up to three
// fields, e.g.:
//   !0 = metadata !{ !"name", !1, 0 }
// The first field is an identity field. It can be any MDString which
// uniquely identifies the type. The second field identifies the type's
// parent node in the tree, or is null or omitted for a root node.
// If the third field is present, it's an integer which if equal to 1
// indicates that the type is "constant".
//
// TODO: The current metadata encoding scheme doesn't support struct
// fields. For example:
//   struct X {
//     double d;
//     int i;
//   };
//   void foo(struct X *x, struct X *y, double *p) {
//     *x = *y;
//     *p = 0.0;
//   }
// Struct X has a double member, so the store to *x can alias the store to *p.
// Currently it's not possible to precisely describe all the things struct X
// aliases, so struct assignments must use conservative TBAA nodes. There's
// no scheme for attaching metadata to @llvm.memcpy yet either.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Module.h"
#include "llvm/Metadata.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

// For testing purposes, enable TBAA only via a special option.
static cl::opt<bool> EnableTBAA("enable-tbaa");

namespace {
  /// TBAANode - This is a simple wrapper around an MDNode which provides a
  /// higher-level interface by hiding the details of how alias analysis
  /// information is encoded in its operands.
  class TBAANode {
    const MDNode *Node;

  public:
    TBAANode() : Node(0) {}
    explicit TBAANode(const MDNode *N) : Node(N) {}

    /// getNode - Get the MDNode for this TBAANode.
    const MDNode *getNode() const { return Node; }

    /// getParent - Get this TBAANode's Alias tree parent.
    TBAANode getParent() const {
      if (Node->getNumOperands() < 2)
        return TBAANode();
      MDNode *P = dyn_cast_or_null<MDNode>(Node->getOperand(1));
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
    TypeBasedAliasAnalysis() : ImmutablePass(ID) {
      initializeTypeBasedAliasAnalysisPass(*PassRegistry::getPassRegistry());
    }

    virtual void initializePass() {
      InitializeAliasAnalysis(this);
    }

    /// getAdjustedAnalysisPointer - This method is used when a pass implements
    /// an analysis interface through multiple inheritance.  If needed, it
    /// should override this to adjust the this pointer as needed for the
    /// specified pass info.
    virtual void *getAdjustedAnalysisPointer(const void *PI) {
      if (PI == &AliasAnalysis::ID)
        return (AliasAnalysis*)this;
      return this;
    }

    bool Aliases(const MDNode *A, const MDNode *B) const;

  private:
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual AliasResult alias(const Location &LocA, const Location &LocB);
    virtual bool pointsToConstantMemory(const Location &Loc);
  };
}  // End of anonymous namespace

// Register this pass...
char TypeBasedAliasAnalysis::ID = 0;
INITIALIZE_AG_PASS(TypeBasedAliasAnalysis, AliasAnalysis, "tbaa",
                   "Type-Based Alias Analysis", false, true, false)

ImmutablePass *llvm::createTypeBasedAliasAnalysisPass() {
  return new TypeBasedAliasAnalysis();
}

void
TypeBasedAliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AliasAnalysis::getAnalysisUsage(AU);
}

/// Aliases - Test whether the type represented by A may alias the
/// type represented by B.
bool
TypeBasedAliasAnalysis::Aliases(const MDNode *A,
                                const MDNode *B) const {
  // Keep track of the root node for A and B.
  TBAANode RootA, RootB;

  // Climb the tree from A to see if we reach B.
  for (TBAANode T(A); ; ) {
    if (T.getNode() == B)
      // B is an ancestor of A.
      return true;

    RootA = T;
    T = T.getParent();
    if (!T.getNode())
      break;
  }

  // Climb the tree from B to see if we reach A.
  for (TBAANode T(B); ; ) {
    if (T.getNode() == A)
      // A is an ancestor of B.
      return true;

    RootB = T;
    T = T.getParent();
    if (!T.getNode())
      break;
  }

  // Neither node is an ancestor of the other.
  
  // If they have different roots, they're part of different potentially
  // unrelated type systems, so we must be conservative.
  if (RootA.getNode() != RootB.getNode())
    return true;

  // If they have the same root, then we've proved there's no alias.
  return false;
}

AliasAnalysis::AliasResult
TypeBasedAliasAnalysis::alias(const Location &LocA,
                              const Location &LocB) {
  if (!EnableTBAA)
    return AliasAnalysis::alias(LocA, LocB);

  // Get the attached MDNodes. If either value lacks a tbaa MDNode, we must
  // be conservative.
  const MDNode *AM = LocA.TBAATag;
  if (!AM) return AliasAnalysis::alias(LocA, LocB);
  const MDNode *BM = LocB.TBAATag;
  if (!BM) return AliasAnalysis::alias(LocA, LocB);

  // If they may alias, chain to the next AliasAnalysis.
  if (Aliases(AM, BM))
    return AliasAnalysis::alias(LocA, LocB);

  // Otherwise return a definitive result.
  return NoAlias;
}

bool TypeBasedAliasAnalysis::pointsToConstantMemory(const Location &Loc) {
  if (!EnableTBAA)
    return AliasAnalysis::pointsToConstantMemory(Loc);

  const MDNode *M = Loc.TBAATag;
  if (!M) return false;

  // If this is an "immutable" type, we can assume the pointer is pointing
  // to constant memory.
  if (TBAANode(M).TypeIsImmutable())
    return true;

  return AliasAnalysis::pointsToConstantMemory(Loc);
}
