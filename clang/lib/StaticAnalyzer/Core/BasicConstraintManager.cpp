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
#include "clang/StaticAnalyzer/Core/PathSensitive/APSIntType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
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
    : SimpleConstraintManager(subengine, statemgr.getBasicVals()),
      ISetFactory(statemgr.getAllocator()) {}

  ProgramStateRef assumeSymEquality(ProgramStateRef State, SymbolRef Sym,
                                    const llvm::APSInt &V,
                                    const llvm::APSInt &Adjustment,
                                    bool Assumption);

  ProgramStateRef assumeSymNE(ProgramStateRef State, SymbolRef Sym,
                              const llvm::APSInt &V,
                              const llvm::APSInt &Adjustment) {
    return assumeSymEquality(State, Sym, V, Adjustment, false);
  }

  ProgramStateRef assumeSymEQ(ProgramStateRef State, SymbolRef Sym,
                              const llvm::APSInt &V,
                              const llvm::APSInt &Adjustment) {
    return assumeSymEquality(State, Sym, V, Adjustment, true);
  }

  ProgramStateRef assumeSymLT(ProgramStateRef state,
                                  SymbolRef sym,
                                  const llvm::APSInt& V,
                                  const llvm::APSInt& Adjustment);

  ProgramStateRef assumeSymGT(ProgramStateRef state,
                                  SymbolRef sym,
                                  const llvm::APSInt& V,
                                  const llvm::APSInt& Adjustment);

  ProgramStateRef assumeSymGE(ProgramStateRef state,
                                  SymbolRef sym,
                                  const llvm::APSInt& V,
                                  const llvm::APSInt& Adjustment);

  ProgramStateRef assumeSymLE(ProgramStateRef state,
                                  SymbolRef sym,
                                  const llvm::APSInt& V,
                                  const llvm::APSInt& Adjustment);

  ProgramStateRef AddEQ(ProgramStateRef state,
                            SymbolRef sym,
                            const llvm::APSInt& V);

  ProgramStateRef AddNE(ProgramStateRef state,
                            SymbolRef sym,
                            const llvm::APSInt& V);

  const llvm::APSInt* getSymVal(ProgramStateRef state,
                                SymbolRef sym) const;

  bool isNotEqual(ProgramStateRef state,
                  SymbolRef sym,
                  const llvm::APSInt& V) const;

  bool isEqual(ProgramStateRef state,
               SymbolRef sym,
               const llvm::APSInt& V) const;

  ProgramStateRef removeDeadBindings(ProgramStateRef state,
                                         SymbolReaper& SymReaper);

  bool performTest(llvm::APSInt SymVal, llvm::APSInt Adjustment,
                   BinaryOperator::Opcode Op, llvm::APSInt ComparisonVal);

  void print(ProgramStateRef state,
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

// FIXME: This is a more general utility and should live somewhere else.
bool BasicConstraintManager::performTest(llvm::APSInt SymVal,
                                         llvm::APSInt Adjustment,
                                         BinaryOperator::Opcode Op,
                                         llvm::APSInt ComparisonVal) {
  APSIntType Type(Adjustment);
  Type.apply(SymVal);
  Type.apply(ComparisonVal);
  SymVal += Adjustment;

  assert(BinaryOperator::isComparisonOp(Op));
  BasicValueFactory &BVF = getBasicVals();
  const llvm::APSInt *Result = BVF.evalAPSInt(Op, SymVal, ComparisonVal);
  assert(Result && "Comparisons should always have valid results.");

  return Result->getBoolValue();
}

ProgramStateRef
BasicConstraintManager::assumeSymEquality(ProgramStateRef State, SymbolRef Sym,
                                          const llvm::APSInt &V,
                                          const llvm::APSInt &Adjustment,
                                          bool Assumption) {
  // Before we do any real work, see if the value can even show up.
  APSIntType AdjustmentType(Adjustment);
  if (AdjustmentType.testInRange(V) != APSIntType::RTR_Within)
    return Assumption ? NULL : State;

  // Get the symbol type.
  BasicValueFactory &BVF = getBasicVals();
  ASTContext &Ctx = BVF.getContext();
  APSIntType SymbolType = BVF.getAPSIntType(Sym->getType(Ctx));

  // First, see if the adjusted value is within range for the symbol.
  llvm::APSInt Adjusted = AdjustmentType.convert(V) - Adjustment;
  if (SymbolType.testInRange(Adjusted) != APSIntType::RTR_Within)
    return Assumption ? NULL : State;

  // Now we can do things properly in the symbol space.
  SymbolType.apply(Adjusted);

  // Second, determine if sym == X, where X+Adjustment != V.
  if (const llvm::APSInt *X = getSymVal(State, Sym)) {
    bool IsFeasible = (*X == Adjusted);
    return (IsFeasible == Assumption) ? State : NULL;
  }

  // Third, determine if we already know sym+Adjustment != V.
  if (isNotEqual(State, Sym, Adjusted))
    return Assumption ? NULL : State;

  // If we reach here, sym is not a constant and we don't know if it is != V.
  // Make the correct assumption.
  if (Assumption)
    return AddEQ(State, Sym, Adjusted);
  else
    return AddNE(State, Sym, Adjusted);
}

// The logic for these will be handled in another ConstraintManager.
// Approximate it here anyway by handling some edge cases.
ProgramStateRef 
BasicConstraintManager::assumeSymLT(ProgramStateRef state,
                                    SymbolRef sym,
                                    const llvm::APSInt &V,
                                    const llvm::APSInt &Adjustment) {
  APSIntType ComparisonType(V), AdjustmentType(Adjustment);

  // Is 'V' out of range above the type?
  llvm::APSInt Max = AdjustmentType.getMaxValue();
  if (V > ComparisonType.convert(Max)) {
    // This path is trivially feasible.
    return state;
  }

  // Is 'V' the smallest possible value, or out of range below the type?
  llvm::APSInt Min = AdjustmentType.getMinValue();
  if (V <= ComparisonType.convert(Min)) {
    // sym cannot be any value less than 'V'.  This path is infeasible.
    return NULL;
  }

  // Reject a path if the value of sym is a constant X and !(X+Adj < V).
  if (const llvm::APSInt *X = getSymVal(state, sym)) {
    bool isFeasible = performTest(*X, Adjustment, BO_LT, V);
    return isFeasible ? state : NULL;
  }

  // FIXME: For now have assuming x < y be the same as assuming sym != V;
  return assumeSymNE(state, sym, V, Adjustment);
}

ProgramStateRef 
BasicConstraintManager::assumeSymGT(ProgramStateRef state,
                                    SymbolRef sym,
                                    const llvm::APSInt &V,
                                    const llvm::APSInt &Adjustment) {
  APSIntType ComparisonType(V), AdjustmentType(Adjustment);

  // Is 'V' the largest possible value, or out of range above the type?
  llvm::APSInt Max = AdjustmentType.getMaxValue();
  if (V >= ComparisonType.convert(Max)) {
    // sym cannot be any value greater than 'V'.  This path is infeasible.
    return NULL;
  }

  // Is 'V' out of range below the type?
  llvm::APSInt Min = AdjustmentType.getMinValue();
  if (V < ComparisonType.convert(Min)) {
    // This path is trivially feasible.
    return state;
  }

  // Reject a path if the value of sym is a constant X and !(X+Adj > V).
  if (const llvm::APSInt *X = getSymVal(state, sym)) {
    bool isFeasible = performTest(*X, Adjustment, BO_GT, V);
    return isFeasible ? state : NULL;
  }

  // FIXME: For now have assuming x > y be the same as assuming sym != V;
  return assumeSymNE(state, sym, V, Adjustment);
}

ProgramStateRef 
BasicConstraintManager::assumeSymGE(ProgramStateRef state,
                                    SymbolRef sym,
                                    const llvm::APSInt &V,
                                    const llvm::APSInt &Adjustment) {
  APSIntType ComparisonType(V), AdjustmentType(Adjustment);

  // Is 'V' the largest possible value, or out of range above the type?
  llvm::APSInt Max = AdjustmentType.getMaxValue();
  ComparisonType.apply(Max);

  if (V > Max) {
    // sym cannot be any value greater than 'V'.  This path is infeasible.
    return NULL;
  } else if (V == Max) {
    // If the path is feasible then as a consequence we know that
    // 'sym+Adjustment == V' because there are no larger values.
    // Add this constraint.
    return assumeSymEQ(state, sym, V, Adjustment);
  }

  // Is 'V' out of range below the type?
  llvm::APSInt Min = AdjustmentType.getMinValue();
  if (V < ComparisonType.convert(Min)) {
    // This path is trivially feasible.
    return state;
  }

  // Reject a path if the value of sym is a constant X and !(X+Adj >= V).
  if (const llvm::APSInt *X = getSymVal(state, sym)) {
    bool isFeasible = performTest(*X, Adjustment, BO_GE, V);
    return isFeasible ? state : NULL;
  }

  return state;
}

ProgramStateRef 
BasicConstraintManager::assumeSymLE(ProgramStateRef state,
                                    SymbolRef sym,
                                    const llvm::APSInt &V,
                                    const llvm::APSInt &Adjustment) {
  APSIntType ComparisonType(V), AdjustmentType(Adjustment);

  // Is 'V' out of range above the type?
  llvm::APSInt Max = AdjustmentType.getMaxValue();
  if (V > ComparisonType.convert(Max)) {
    // This path is trivially feasible.
    return state;
  }

  // Is 'V' the smallest possible value, or out of range below the type?
  llvm::APSInt Min = AdjustmentType.getMinValue();
  ComparisonType.apply(Min);

  if (V < Min) {
    // sym cannot be any value less than 'V'.  This path is infeasible.
    return NULL;
  } else if (V == Min) {
    // If the path is feasible then as a consequence we know that
    // 'sym+Adjustment == V' because there are no smaller values.
    // Add this constraint.
    return assumeSymEQ(state, sym, V, Adjustment);
  }

  // Reject a path if the value of sym is a constant X and !(X+Adj >= V).
  if (const llvm::APSInt *X = getSymVal(state, sym)) {
    bool isFeasible = performTest(*X, Adjustment, BO_LE, V);
    return isFeasible ? state : NULL;
  }

  return state;
}

ProgramStateRef BasicConstraintManager::AddEQ(ProgramStateRef state,
                                                  SymbolRef sym,
                                             const llvm::APSInt& V) {
  // Create a new state with the old binding replaced.
  return state->set<ConstEq>(sym, &getBasicVals().getValue(V));
}

ProgramStateRef BasicConstraintManager::AddNE(ProgramStateRef state,
                                                  SymbolRef sym,
                                                  const llvm::APSInt& V) {

  // First, retrieve the NE-set associated with the given symbol.
  ConstNotEqTy::data_type* T = state->get<ConstNotEq>(sym);
  ProgramState::IntSetTy S = T ? *T : ISetFactory.getEmptySet();

  // Now add V to the NE set.
  S = ISetFactory.add(S, &getBasicVals().getValue(V));

  // Create a new state with the old binding replaced.
  return state->set<ConstNotEq>(sym, S);
}

const llvm::APSInt* BasicConstraintManager::getSymVal(ProgramStateRef state,
                                                      SymbolRef sym) const {
  const ConstEqTy::data_type* T = state->get<ConstEq>(sym);
  return T ? *T : NULL;
}

bool BasicConstraintManager::isNotEqual(ProgramStateRef state,
                                        SymbolRef sym,
                                        const llvm::APSInt& V) const {

  // Retrieve the NE-set associated with the given symbol.
  const ConstNotEqTy::data_type* T = state->get<ConstNotEq>(sym);

  // See if V is present in the NE-set.
  return T ? T->contains(&getBasicVals().getValue(V)) : false;
}

bool BasicConstraintManager::isEqual(ProgramStateRef state,
                                     SymbolRef sym,
                                     const llvm::APSInt& V) const {
  // Retrieve the EQ-set associated with the given symbol.
  const ConstEqTy::data_type* T = state->get<ConstEq>(sym);
  // See if V is present in the EQ-set.
  return T ? **T == V : false;
}

/// Scan all symbols referenced by the constraints. If the symbol is not alive
/// as marked in LSymbols, mark it as dead in DSymbols.
ProgramStateRef 
BasicConstraintManager::removeDeadBindings(ProgramStateRef state,
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

void BasicConstraintManager::print(ProgramStateRef state,
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
