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
//  equality and inequality constraints on symbolic values of ProgramState.
//
//===----------------------------------------------------------------------===//

#include "SimpleConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/TransferFuncs.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;


namespace { class ConstNotEq {}; }
namespace { class ConstEq {}; }

typedef llvm::ImmutableMap<SymbolRef,ProgramState::IntSetTy> ConstNotEqTy;
typedef llvm::ImmutableMap<SymbolRef,const llvm::APSInt*> ConstEqTy;

static int ConstEqIndex = 0;
static int ConstNotEqIndex = 0;

namespace clang {
namespace ento {
template<>
struct ProgramStateTrait<ConstNotEq> :
  public ProgramStatePartialTrait<ConstNotEqTy> {
  static inline void *GDMIndex() { return &ConstNotEqIndex; }
};

template<>
struct ProgramStateTrait<ConstEq> : public ProgramStatePartialTrait<ConstEqTy> {
  static inline void *GDMIndex() { return &ConstEqIndex; }
};
}
}

namespace {
// BasicConstraintManager only tracks equality and inequality constraints of
// constants and integer variables.
class BasicConstraintManager
  : public SimpleConstraintManager {
  ProgramState::IntSetTy::Factory ISetFactory;
public:
  BasicConstraintManager(ProgramStateManager &statemgr, SubEngine &subengine)
    : SimpleConstraintManager(subengine), 
      ISetFactory(statemgr.getAllocator()) {}

  const ProgramState *assumeSymNE(const ProgramState *state,
                                  SymbolRef sym,
                                  const llvm::APSInt& V,
                                  const llvm::APSInt& Adjustment);

  const ProgramState *assumeSymEQ(const ProgramState *state,
                                  SymbolRef sym,
                                  const llvm::APSInt& V,
                                  const llvm::APSInt& Adjustment);

  const ProgramState *assumeSymLT(const ProgramState *state,
                                  SymbolRef sym,
                                  const llvm::APSInt& V,
                                  const llvm::APSInt& Adjustment);

  const ProgramState *assumeSymGT(const ProgramState *state,
                                  SymbolRef sym,
                                  const llvm::APSInt& V,
                                  const llvm::APSInt& Adjustment);

  const ProgramState *assumeSymGE(const ProgramState *state,
                                  SymbolRef sym,
                                  const llvm::APSInt& V,
                                  const llvm::APSInt& Adjustment);

  const ProgramState *assumeSymLE(const ProgramState *state,
                                  SymbolRef sym,
                                  const llvm::APSInt& V,
                                  const llvm::APSInt& Adjustment);

  const ProgramState *AddEQ(const ProgramState *state,
                            SymbolRef sym,
                            const llvm::APSInt& V);

  const ProgramState *AddNE(const ProgramState *state,
                            SymbolRef sym,
                            const llvm::APSInt& V);

  const llvm::APSInt* getSymVal(const ProgramState *state,
                                SymbolRef sym) const;

  bool isNotEqual(const ProgramState *state,
                  SymbolRef sym,
                  const llvm::APSInt& V) const;

  bool isEqual(const ProgramState *state,
               SymbolRef sym,
               const llvm::APSInt& V) const;

  const ProgramState *removeDeadBindings(const ProgramState *state,
                                         SymbolReaper& SymReaper);

  void print(const ProgramState *state,
             raw_ostream &Out,
             const char* nl,
             const char *sep);
};

} // end anonymous namespace

ConstraintManager*
ento::CreateBasicConstraintManager(ProgramStateManager& statemgr,
                                   SubEngine &subengine) {
  return new BasicConstraintManager(statemgr, subengine);
}

const ProgramState*
BasicConstraintManager::assumeSymNE(const ProgramState *state,
                                    SymbolRef sym,
                                    const llvm::APSInt &V,
                                    const llvm::APSInt &Adjustment) {
  // First, determine if sym == X, where X+Adjustment != V.
  llvm::APSInt Adjusted = V-Adjustment;
  if (const llvm::APSInt* X = getSymVal(state, sym)) {
    bool isFeasible = (*X != Adjusted);
    return isFeasible ? state : NULL;
  }

  // Second, determine if sym+Adjustment != V.
  if (isNotEqual(state, sym, Adjusted))
    return state;

  // If we reach here, sym is not a constant and we don't know if it is != V.
  // Make that assumption.
  return AddNE(state, sym, Adjusted);
}

const ProgramState*
BasicConstraintManager::assumeSymEQ(const ProgramState *state,
                                    SymbolRef sym,
                                    const llvm::APSInt &V,
                                    const llvm::APSInt &Adjustment) {
  // First, determine if sym == X, where X+Adjustment != V.
  llvm::APSInt Adjusted = V-Adjustment;
  if (const llvm::APSInt* X = getSymVal(state, sym)) {
    bool isFeasible = (*X == Adjusted);
    return isFeasible ? state : NULL;
  }

  // Second, determine if sym+Adjustment != V.
  if (isNotEqual(state, sym, Adjusted))
    return NULL;

  // If we reach here, sym is not a constant and we don't know if it is == V.
  // Make that assumption.
  return AddEQ(state, sym, Adjusted);
}

// The logic for these will be handled in another ConstraintManager.
const ProgramState*
BasicConstraintManager::assumeSymLT(const ProgramState *state,
                                    SymbolRef sym,
                                    const llvm::APSInt &V,
                                    const llvm::APSInt &Adjustment) {
  // Is 'V' the smallest possible value?
  if (V == llvm::APSInt::getMinValue(V.getBitWidth(), V.isUnsigned())) {
    // sym cannot be any value less than 'V'.  This path is infeasible.
    return NULL;
  }

  // FIXME: For now have assuming x < y be the same as assuming sym != V;
  return assumeSymNE(state, sym, V, Adjustment);
}

const ProgramState*
BasicConstraintManager::assumeSymGT(const ProgramState *state,
                                    SymbolRef sym,
                                    const llvm::APSInt &V,
                                    const llvm::APSInt &Adjustment) {
  // Is 'V' the largest possible value?
  if (V == llvm::APSInt::getMaxValue(V.getBitWidth(), V.isUnsigned())) {
    // sym cannot be any value greater than 'V'.  This path is infeasible.
    return NULL;
  }

  // FIXME: For now have assuming x > y be the same as assuming sym != V;
  return assumeSymNE(state, sym, V, Adjustment);
}

const ProgramState*
BasicConstraintManager::assumeSymGE(const ProgramState *state,
                                    SymbolRef sym,
                                    const llvm::APSInt &V,
                                    const llvm::APSInt &Adjustment) {
  // Reject a path if the value of sym is a constant X and !(X+Adj >= V).
  if (const llvm::APSInt *X = getSymVal(state, sym)) {
    bool isFeasible = (*X >= V-Adjustment);
    return isFeasible ? state : NULL;
  }

  // Sym is not a constant, but it is worth looking to see if V is the
  // maximum integer value.
  if (V == llvm::APSInt::getMaxValue(V.getBitWidth(), V.isUnsigned())) {
    llvm::APSInt Adjusted = V-Adjustment;

    // If we know that sym != V (after adjustment), then this condition
    // is infeasible since there is no other value greater than V.
    bool isFeasible = !isNotEqual(state, sym, Adjusted);

    // If the path is still feasible then as a consequence we know that
    // 'sym+Adjustment == V' because there are no larger values.
    // Add this constraint.
    return isFeasible ? AddEQ(state, sym, Adjusted) : NULL;
  }

  return state;
}

const ProgramState*
BasicConstraintManager::assumeSymLE(const ProgramState *state,
                                    SymbolRef sym,
                                    const llvm::APSInt &V,
                                    const llvm::APSInt &Adjustment) {
  // Reject a path if the value of sym is a constant X and !(X+Adj <= V).
  if (const llvm::APSInt* X = getSymVal(state, sym)) {
    bool isFeasible = (*X <= V-Adjustment);
    return isFeasible ? state : NULL;
  }

  // Sym is not a constant, but it is worth looking to see if V is the
  // minimum integer value.
  if (V == llvm::APSInt::getMinValue(V.getBitWidth(), V.isUnsigned())) {
    llvm::APSInt Adjusted = V-Adjustment;

    // If we know that sym != V (after adjustment), then this condition
    // is infeasible since there is no other value less than V.
    bool isFeasible = !isNotEqual(state, sym, Adjusted);

    // If the path is still feasible then as a consequence we know that
    // 'sym+Adjustment == V' because there are no smaller values.
    // Add this constraint.
    return isFeasible ? AddEQ(state, sym, Adjusted) : NULL;
  }

  return state;
}

const ProgramState *BasicConstraintManager::AddEQ(const ProgramState *state,
                                                  SymbolRef sym,
                                             const llvm::APSInt& V) {
  // Create a new state with the old binding replaced.
  return state->set<ConstEq>(sym, &state->getBasicVals().getValue(V));
}

const ProgramState *BasicConstraintManager::AddNE(const ProgramState *state,
                                                  SymbolRef sym,
                                                  const llvm::APSInt& V) {

  // First, retrieve the NE-set associated with the given symbol.
  ConstNotEqTy::data_type* T = state->get<ConstNotEq>(sym);
  ProgramState::IntSetTy S = T ? *T : ISetFactory.getEmptySet();

  // Now add V to the NE set.
  S = ISetFactory.add(S, &state->getBasicVals().getValue(V));

  // Create a new state with the old binding replaced.
  return state->set<ConstNotEq>(sym, S);
}

const llvm::APSInt* BasicConstraintManager::getSymVal(const ProgramState *state,
                                                      SymbolRef sym) const {
  const ConstEqTy::data_type* T = state->get<ConstEq>(sym);
  return T ? *T : NULL;
}

bool BasicConstraintManager::isNotEqual(const ProgramState *state,
                                        SymbolRef sym,
                                        const llvm::APSInt& V) const {

  // Retrieve the NE-set associated with the given symbol.
  const ConstNotEqTy::data_type* T = state->get<ConstNotEq>(sym);

  // See if V is present in the NE-set.
  return T ? T->contains(&state->getBasicVals().getValue(V)) : false;
}

bool BasicConstraintManager::isEqual(const ProgramState *state,
                                     SymbolRef sym,
                                     const llvm::APSInt& V) const {
  // Retrieve the EQ-set associated with the given symbol.
  const ConstEqTy::data_type* T = state->get<ConstEq>(sym);
  // See if V is present in the EQ-set.
  return T ? **T == V : false;
}

/// Scan all symbols referenced by the constraints. If the symbol is not alive
/// as marked in LSymbols, mark it as dead in DSymbols.
const ProgramState*
BasicConstraintManager::removeDeadBindings(const ProgramState *state,
                                           SymbolReaper& SymReaper) {

  ConstEqTy CE = state->get<ConstEq>();
  ConstEqTy::Factory& CEFactory = state->get_context<ConstEq>();

  for (ConstEqTy::iterator I = CE.begin(), E = CE.end(); I!=E; ++I) {
    SymbolRef sym = I.getKey();
    if (SymReaper.maybeDead(sym))
      CE = CEFactory.remove(CE, sym);
  }
  state = state->set<ConstEq>(CE);

  ConstNotEqTy CNE = state->get<ConstNotEq>();
  ConstNotEqTy::Factory& CNEFactory = state->get_context<ConstNotEq>();

  for (ConstNotEqTy::iterator I = CNE.begin(), E = CNE.end(); I != E; ++I) {
    SymbolRef sym = I.getKey();
    if (SymReaper.maybeDead(sym))
      CNE = CNEFactory.remove(CNE, sym);
  }

  return state->set<ConstNotEq>(CNE);
}

void BasicConstraintManager::print(const ProgramState *state,
                                   raw_ostream &Out,
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

      ProgramState::IntSetTy::iterator J = I.getData().begin(),
                                  EJ = I.getData().end();

      for ( ; J != EJ; ++J) {
        if (isFirst) isFirst = false;
        else Out << ", ";

        Out << (*J)->getSExtValue(); // Hack: should print to raw_ostream.
      }
    }
  }
}
