//===- ScopedNoAliasAA.cpp - Scoped No-Alias Alias Analysis ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ScopedNoAlias alias-analysis pass, which implements
// metadata-based scoped no-alias support.
//
// Alias-analysis scopes are defined similar to TBAA nodes:
//
// !scope0 = metadata !{ metadata !"scope of foo()" }
// !scope1 = metadata !{ metadata !"scope 1", metadata !scope0 }
// !scope2 = metadata !{ metadata !"scope 2", metadata !scope0 }
// !scope3 = metadata !{ metadata !"scope 2.1", metadata !scope2 }
// !scope4 = metadata !{ metadata !"scope 2.2", metadata !scope2 }
//
// Loads and stores can be tagged with an alias-analysis scope, and also, with
// a noalias tag for a specific scope:
//
// ... = load %ptr1, !alias.scope !{ !scope1 }
// ... = load %ptr2, !alias.scope !{ !scope1, !scope2 }, !noalias !{ !scope1 }
//
// When evaluating an aliasing query, if one of the instructions is associated
// with an alias.scope id that is identical to the noalias scope associated
// with the other instruction, or is a descendant (in the scope hierarchy) of
// the noalias scope associated with the other instruction, then the two memory
// accesses are assumed not to alias.
//
// Note that if the first element of the scope metadata is a string, then it
// can be combined accross functions and translation units. The string can be
// replaced by a self-reference to create globally unqiue scope identifiers.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

// A handy option for disabling scoped no-alias functionality. The same effect
// can also be achieved by stripping the associated metadata tags from IR, but
// this option is sometimes more convenient.
static cl::opt<bool>
EnableScopedNoAlias("enable-scoped-noalias", cl::init(true));

namespace {
/// AliasScopeNode - This is a simple wrapper around an MDNode which provides
/// a higher-level interface by hiding the details of how alias analysis
/// information is encoded in its operands.
class AliasScopeNode {
  const MDNode *Node;

public:
  AliasScopeNode() : Node(0) {}
  explicit AliasScopeNode(const MDNode *N) : Node(N) {}

  /// getNode - Get the MDNode for this AliasScopeNode.
  const MDNode *getNode() const { return Node; }

  /// getParent - Get this AliasScopeNode's Alias tree parent.
  AliasScopeNode getParent() const {
    if (Node->getNumOperands() < 2)
      return AliasScopeNode();
    MDNode *P = dyn_cast_or_null<MDNode>(Node->getOperand(1));
    if (!P)
      return AliasScopeNode();
    // Ok, this node has a valid parent. Return it.
    return AliasScopeNode(P);
  }
};

/// ScopedNoAliasAA - This is a simple alias analysis
/// implementation that uses scoped-noalias metadata to answer queries.
class ScopedNoAliasAA : public ImmutablePass, public AliasAnalysis {
public:
  static char ID; // Class identification, replacement for typeinfo
  ScopedNoAliasAA() : ImmutablePass(ID) {
    initializeScopedNoAliasAAPass(*PassRegistry::getPassRegistry());
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

protected:
  bool mayAlias(const MDNode *A, const MDNode *B) const;
  bool mayAliasInScopes(const MDNode *Scopes, const MDNode *NoAlias) const;

private:
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual AliasResult alias(const Location &LocA, const Location &LocB);
  virtual bool pointsToConstantMemory(const Location &Loc, bool OrLocal);
  virtual ModRefBehavior getModRefBehavior(ImmutableCallSite CS);
  virtual ModRefBehavior getModRefBehavior(const Function *F);
  virtual ModRefResult getModRefInfo(ImmutableCallSite CS,
                                     const Location &Loc);
  virtual ModRefResult getModRefInfo(ImmutableCallSite CS1,
                                     ImmutableCallSite CS2);
};
}  // End of anonymous namespace

// Register this pass...
char ScopedNoAliasAA::ID = 0;
INITIALIZE_AG_PASS(ScopedNoAliasAA, AliasAnalysis, "scoped-noalias",
                   "Scoped NoAlias Alias Analysis", false, true, false)

ImmutablePass *llvm::createScopedNoAliasAAPass() {
  return new ScopedNoAliasAA();
}

void
ScopedNoAliasAA::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AliasAnalysis::getAnalysisUsage(AU);
}

/// mayAlias - Test whether the scope represented by A may alias the
/// scope represented by B. Specifically, A is the target scope, and B is the
/// noalias scope.
bool
ScopedNoAliasAA::mayAlias(const MDNode *A,
                          const MDNode *B) const {
  // Climb the tree from A to see if we reach B.
  for (AliasScopeNode T(A); ; ) {
    if (T.getNode() == B)
      // B is an ancestor of A.
      return false;

    T = T.getParent();
    if (!T.getNode())
      break;
  }

  return true;
}

bool
ScopedNoAliasAA::mayAliasInScopes(const MDNode *Scopes,
                                  const MDNode *NoAlias) const {
  if (!Scopes || !NoAlias)
    return true;

  for (unsigned i = 0, ie = Scopes->getNumOperands(); i != ie; ++i)
    if (const MDNode *SMD = dyn_cast<MDNode>(Scopes->getOperand(i)))
      for (unsigned j = 0, je = NoAlias->getNumOperands(); j != je; ++j)
        if (const MDNode *NAMD = dyn_cast<MDNode>(NoAlias->getOperand(j)))
          if (!mayAlias(SMD, NAMD))
            return false;

  return true; 
}

AliasAnalysis::AliasResult
ScopedNoAliasAA::alias(const Location &LocA, const Location &LocB) {
  if (!EnableScopedNoAlias)
    return AliasAnalysis::alias(LocA, LocB);

  // Get the attached MDNodes.
  const MDNode *AScopes = LocA.AATags.Scope,
               *BScopes = LocB.AATags.Scope;

  const MDNode *ANoAlias = LocA.AATags.NoAlias,
               *BNoAlias = LocB.AATags.NoAlias;

  if (!mayAliasInScopes(AScopes, BNoAlias))
    return NoAlias;

  if (!mayAliasInScopes(BScopes, ANoAlias))
    return NoAlias;

  // If they may alias, chain to the next AliasAnalysis.
  return AliasAnalysis::alias(LocA, LocB);
}

bool ScopedNoAliasAA::pointsToConstantMemory(const Location &Loc,
                                             bool OrLocal) {
  return AliasAnalysis::pointsToConstantMemory(Loc, OrLocal);
}

AliasAnalysis::ModRefBehavior
ScopedNoAliasAA::getModRefBehavior(ImmutableCallSite CS) {
  return AliasAnalysis::getModRefBehavior(CS);
}

AliasAnalysis::ModRefBehavior
ScopedNoAliasAA::getModRefBehavior(const Function *F) {
  return AliasAnalysis::getModRefBehavior(F);
}

AliasAnalysis::ModRefResult
ScopedNoAliasAA::getModRefInfo(ImmutableCallSite CS, const Location &Loc) {
  if (!EnableScopedNoAlias)
    return AliasAnalysis::getModRefInfo(CS, Loc);

  if (!mayAliasInScopes(Loc.AATags.Scope,
        CS.getInstruction()->getMetadata(LLVMContext::MD_noalias)))
    return NoModRef;

  if (!mayAliasInScopes(
        CS.getInstruction()->getMetadata(LLVMContext::MD_alias_scope),
        Loc.AATags.NoAlias))
    return NoModRef;

  return AliasAnalysis::getModRefInfo(CS, Loc);
}

AliasAnalysis::ModRefResult
ScopedNoAliasAA::getModRefInfo(ImmutableCallSite CS1, ImmutableCallSite CS2) {
  if (!EnableScopedNoAlias)
    return AliasAnalysis::getModRefInfo(CS1, CS2);

  if (!mayAliasInScopes(
        CS1.getInstruction()->getMetadata(LLVMContext::MD_alias_scope),
        CS2.getInstruction()->getMetadata(LLVMContext::MD_noalias)))
    return NoModRef;

  if (!mayAliasInScopes(
        CS2.getInstruction()->getMetadata(LLVMContext::MD_alias_scope),
        CS1.getInstruction()->getMetadata(LLVMContext::MD_noalias)))
    return NoModRef;

  return AliasAnalysis::getModRefInfo(CS1, CS2);
}

