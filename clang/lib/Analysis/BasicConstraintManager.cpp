//== BasicConstraintManager.cpp - Manage basic constraints.------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines BasicConstraintManager, a class that tracks simple
//  equality and inequality constraints on symbolic values of GRState.
//
//===----------------------------------------------------------------------===//

#include "SimpleConstraintManager.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/PathSensitive/GRStateTrait.h"
#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;


namespace { class VISIBILITY_HIDDEN ConstNotEq {}; }
namespace { class VISIBILITY_HIDDEN ConstEq {}; }

typedef llvm::ImmutableMap<SymbolRef,GRState::IntSetTy> ConstNotEqTy;
typedef llvm::ImmutableMap<SymbolRef,const llvm::APSInt*> ConstEqTy;

static int ConstEqIndex = 0;
static int ConstNotEqIndex = 0;

namespace clang {
template<>
struct GRStateTrait<ConstNotEq> : public GRStatePartialTrait<ConstNotEqTy> {
  static inline void* GDMIndex() { return &ConstNotEqIndex; }
};

template<>
struct GRStateTrait<ConstEq> : public GRStatePartialTrait<ConstEqTy> {
  static inline void* GDMIndex() { return &ConstEqIndex; }
};
}

namespace {
// BasicConstraintManager only tracks equality and inequality constraints of
// constants and integer variables.
class VISIBILITY_HIDDEN BasicConstraintManager
  : public SimpleConstraintManager {
  GRState::IntSetTy::Factory ISetFactory;
public:
  BasicConstraintManager(GRStateManager& statemgr)
    : ISetFactory(statemgr.getAllocator()) {}

  const GRState* AssumeSymNE(const GRState* state, SymbolRef sym,
                             const llvm::APSInt& V);

  const GRState* AssumeSymEQ(const GRState* state, SymbolRef sym,
                             const llvm::APSInt& V);

  const GRState* AssumeSymLT(const GRState* state, SymbolRef sym,
                             const llvm::APSInt& V);

  const GRState* AssumeSymGT(const GRState* state, SymbolRef sym,
                             const llvm::APSInt& V);

  const GRState* AssumeSymGE(const GRState* state, SymbolRef sym,
                             const llvm::APSInt& V);

  const GRState* AssumeSymLE(const GRState* state, SymbolRef sym,
                             const llvm::APSInt& V);

  const GRState* AddEQ(const GRState* state, SymbolRef sym, const llvm::APSInt& V);

  const GRState* AddNE(const GRState* state, SymbolRef sym, const llvm::APSInt& V);

  const llvm::APSInt* getSymVal(const GRState* state, SymbolRef sym) const;
  bool isNotEqual(const GRState* state, SymbolRef sym, const llvm::APSInt& V)
      const;
  bool isEqual(const GRState* state, SymbolRef sym, const llvm::APSInt& V)
      const;

  const GRState* RemoveDeadBindings(const GRState* state, SymbolReaper& SymReaper);

  void print(const GRState* state, llvm::raw_ostream& Out,
             const char* nl, const char *sep);
};

} // end anonymous namespace

ConstraintManager* clang::CreateBasicConstraintManager(GRStateManager& StateMgr)
{
  return new BasicConstraintManager(StateMgr);
}

const GRState*
BasicConstraintManager::AssumeSymNE(const GRState *state, SymbolRef sym,
                                    const llvm::APSInt& V) {
  // First, determine if sym == X, where X != V.
  if (const llvm::APSInt* X = getSymVal(state, sym)) {
    bool isFeasible = (*X != V);
    return isFeasible ? state : NULL;
  }

  // Second, determine if sym != V.
  if (isNotEqual(state, sym, V))
    return state;

  // If we reach here, sym is not a constant and we don't know if it is != V.
  // Make that assumption.
  return AddNE(state, sym, V);
}

const GRState *BasicConstraintManager::AssumeSymEQ(const GRState *state,
                                                   SymbolRef sym,
                                                   const llvm::APSInt &V) {
  // First, determine if sym == X, where X != V.
  if (const llvm::APSInt* X = getSymVal(state, sym)) {
    bool isFeasible = *X == V;
    return isFeasible ? state : NULL;
  }

  // Second, determine if sym != V.
  if (isNotEqual(state, sym, V))
    return NULL;

  // If we reach here, sym is not a constant and we don't know if it is == V.
  // Make that assumption.
  return AddEQ(state, sym, V);
}

// These logic will be handled in another ConstraintManager.
const GRState *BasicConstraintManager::AssumeSymLT(const GRState *state,
                                                   SymbolRef sym,
                                                   const llvm::APSInt& V) {
  // Is 'V' the smallest possible value?
  if (V == llvm::APSInt::getMinValue(V.getBitWidth(), V.isUnsigned())) {
    // sym cannot be any value less than 'V'.  This path is infeasible.
    return NULL;
  }

  // FIXME: For now have assuming x < y be the same as assuming sym != V;
  return AssumeSymNE(state, sym, V);
}

const GRState *BasicConstraintManager::AssumeSymGT(const GRState *state,
                                                   SymbolRef sym,
                                                   const llvm::APSInt& V) {

  // Is 'V' the largest possible value?
  if (V == llvm::APSInt::getMaxValue(V.getBitWidth(), V.isUnsigned())) {
    // sym cannot be any value greater than 'V'.  This path is infeasible.
    return NULL;
  }

  // FIXME: For now have assuming x > y be the same as assuming sym != V;
  return AssumeSymNE(state, sym, V);
}

const GRState *BasicConstraintManager::AssumeSymGE(const GRState *state,
                                                   SymbolRef sym,
                                                   const llvm::APSInt &V) {

  // Reject a path if the value of sym is a constant X and !(X >= V).
  if (const llvm::APSInt *X = getSymVal(state, sym)) {
    bool isFeasible = *X >= V;
    return isFeasible ? state : NULL;
  }

  // Sym is not a constant, but it is worth looking to see if V is the
  // maximum integer value.
  if (V == llvm::APSInt::getMaxValue(V.getBitWidth(), V.isUnsigned())) {
    // If we know that sym != V, then this condition is infeasible since
    // there is no other value greater than V.
    bool isFeasible = !isNotEqual(state, sym, V);

    // If the path is still feasible then as a consequence we know that
    // 'sym == V' because we cannot have 'sym > V' (no larger values).
    // Add this constraint.
    return isFeasible ? AddEQ(state, sym, V) : NULL;
  }

  return state;
}

const GRState*
BasicConstraintManager::AssumeSymLE(const GRState* state, SymbolRef sym,
                                    const llvm::APSInt& V) {

  // Reject a path if the value of sym is a constant X and !(X <= V).
  if (const llvm::APSInt* X = getSymVal(state, sym)) {
    bool isFeasible = *X <= V;
    return isFeasible ? state : NULL;
  }

  // Sym is not a constant, but it is worth looking to see if V is the
  // minimum integer value.
  if (V == llvm::APSInt::getMinValue(V.getBitWidth(), V.isUnsigned())) {
    // If we know that sym != V, then this condition is infeasible since
    // there is no other value less than V.
    bool isFeasible = !isNotEqual(state, sym, V);

    // If the path is still feasible then as a consequence we know that
    // 'sym == V' because we cannot have 'sym < V' (no smaller values).
    // Add this constraint.
    return isFeasible ? AddEQ(state, sym, V) : NULL;
  }

  return state;
}

const GRState* BasicConstraintManager::AddEQ(const GRState* state, SymbolRef sym,
                                             const llvm::APSInt& V) {
  // Create a new state with the old binding replaced.
  return state->set<ConstEq>(sym, &V);
}

const GRState* BasicConstraintManager::AddNE(const GRState* state, SymbolRef sym,
                                             const llvm::APSInt& V) {

  // First, retrieve the NE-set associated with the given symbol.
  ConstNotEqTy::data_type* T = state->get<ConstNotEq>(sym);
  GRState::IntSetTy S = T ? *T : ISetFactory.GetEmptySet();

  // Now add V to the NE set.
  S = ISetFactory.Add(S, &V);

  // Create a new state with the old binding replaced.
  return state->set<ConstNotEq>(sym, S);
}

const llvm::APSInt* BasicConstraintManager::getSymVal(const GRState* state,
                                                      SymbolRef sym) const {
  const ConstEqTy::data_type* T = state->get<ConstEq>(sym);
  return T ? *T : NULL;
}

bool BasicConstraintManager::isNotEqual(const GRState* state, SymbolRef sym,
                                        const llvm::APSInt& V) const {

  // Retrieve the NE-set associated with the given symbol.
  const ConstNotEqTy::data_type* T = state->get<ConstNotEq>(sym);

  // See if V is present in the NE-set.
  return T ? T->contains(&V) : false;
}

bool BasicConstraintManager::isEqual(const GRState* state, SymbolRef sym,
                                     const llvm::APSInt& V) const {
  // Retrieve the EQ-set associated with the given symbol.
  const ConstEqTy::data_type* T = state->get<ConstEq>(sym);
  // See if V is present in the EQ-set.
  return T ? **T == V : false;
}

/// Scan all symbols referenced by the constraints. If the symbol is not alive
/// as marked in LSymbols, mark it as dead in DSymbols.
const GRState*
BasicConstraintManager::RemoveDeadBindings(const GRState* state,
                                           SymbolReaper& SymReaper) {

  ConstEqTy CE = state->get<ConstEq>();
  ConstEqTy::Factory& CEFactory = state->get_context<ConstEq>();

  for (ConstEqTy::iterator I = CE.begin(), E = CE.end(); I!=E; ++I) {
    SymbolRef sym = I.getKey();
    if (SymReaper.maybeDead(sym)) CE = CEFactory.Remove(CE, sym);
  }
  state = state->set<ConstEq>(CE);

  ConstNotEqTy CNE = state->get<ConstNotEq>();
  ConstNotEqTy::Factory& CNEFactory = state->get_context<ConstNotEq>();

  for (ConstNotEqTy::iterator I = CNE.begin(), E = CNE.end(); I != E; ++I) {
    SymbolRef sym = I.getKey();
    if (SymReaper.maybeDead(sym)) CNE = CNEFactory.Remove(CNE, sym);
  }

  return state->set<ConstNotEq>(CNE);
}

void BasicConstraintManager::print(const GRState* state, llvm::raw_ostream& Out,
                                   const char* nl, const char *sep) {
  // Print equality constraints.

  ConstEqTy CE = state->get<ConstEq>();

  if (!CE.isEmpty()) {
    Out << nl << sep << "'==' constraints:";
    for (ConstEqTy::iterator I = CE.begin(), E = CE.end(); I!=E; ++I)
      Out << nl << " $" << I.getKey() << " : " << *I.getData();
  }

  // Print != constraints.

  ConstNotEqTy CNE = state->get<ConstNotEq>();

  if (!CNE.isEmpty()) {
    Out << nl << sep << "'!=' constraints:";

    for (ConstNotEqTy::iterator I = CNE.begin(), EI = CNE.end(); I!=EI; ++I) {
      Out << nl << " $" << I.getKey() << " : ";
      bool isFirst = true;

      GRState::IntSetTy::iterator J = I.getData().begin(),
                                  EJ = I.getData().end();

      for ( ; J != EJ; ++J) {
        if (isFirst) isFirst = false;
        else Out << ", ";

        Out << (*J)->getSExtValue(); // Hack: should print to raw_ostream.
      }
    }
  }
}
