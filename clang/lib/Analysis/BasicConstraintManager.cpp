#include "clang/Analysis/PathSensitive/ConstraintManager.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

namespace {

// BasicConstraintManager only tracks equality and inequality constraints of
// constants and integer variables.
class VISIBILITY_HIDDEN BasicConstraintManager : public ConstraintManager {
  typedef llvm::ImmutableMap<SymbolID, GRState::IntSetTy> ConstNotEqTy;
  typedef llvm::ImmutableMap<SymbolID, const llvm::APSInt*> ConstEqTy;

  GRStateManager& StateMgr;

public:
  BasicConstraintManager(GRStateManager& statemgr) : StateMgr(statemgr) {}

  virtual const GRState* Assume(const GRState* St, RVal Cond,
                                bool Assumption, bool& isFeasible);

  const GRState* Assume(const GRState* St, LVal Cond, bool Assumption,
                        bool& isFeasible);

  const GRState* AssumeAux(const GRState* St, LVal Cond,bool Assumption,
                           bool& isFeasible);

  const GRState* Assume(const GRState* St, NonLVal Cond, bool Assumption,
                        bool& isFeasible);

  const GRState* AssumeAux(const GRState* St, NonLVal Cond, bool Assumption,
                           bool& isFeasible);

  const GRState* AssumeSymInt(const GRState* St, bool Assumption,
                              const SymIntConstraint& C, bool& isFeasible);

  const GRState* AssumeSymNE(const GRState* St, SymbolID sym,
                                const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymEQ(const GRState* St, SymbolID sym,
                                const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymLT(const GRState* St, SymbolID sym,
                                    const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymGT(const GRState* St, SymbolID sym,
                             const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymGE(const GRState* St, SymbolID sym,
                             const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymLE(const GRState* St, SymbolID sym,
                             const llvm::APSInt& V, bool& isFeasible);
  };

} // end anonymous namespace

ConstraintManager* clang::CreateBasicConstraintManager(GRStateManager& StateMgr)
{
  return new BasicConstraintManager(StateMgr);
}

const GRState* BasicConstraintManager::Assume(const GRState* St, RVal Cond,
                                            bool Assumption, bool& isFeasible) {
  if (Cond.isUnknown()) {
    isFeasible = true;
    return St;
  }

  if (isa<NonLVal>(Cond))
    return Assume(St, cast<NonLVal>(Cond), Assumption, isFeasible);
  else
    return Assume(St, cast<LVal>(Cond), Assumption, isFeasible);
}

const GRState* BasicConstraintManager::Assume(const GRState* St, LVal Cond,
                                            bool Assumption, bool& isFeasible) {
  St = AssumeAux(St, Cond, Assumption, isFeasible);
  // TF->EvalAssume(*this, St, Cond, Assumption, isFeasible)
  return St;
}

const GRState* BasicConstraintManager::AssumeAux(const GRState* St, LVal Cond,
                                            bool Assumption, bool& isFeasible) {
  BasicValueFactory& BasicVals = StateMgr.getBasicVals();

  switch (Cond.getSubKind()) {
  default:
    assert (false && "'Assume' not implemented for this LVal.");
    return St;

  case lval::SymbolValKind:
    if (Assumption)
      return AssumeSymNE(St, cast<lval::SymbolVal>(Cond).getSymbol(),
                         BasicVals.getZeroWithPtrWidth(), isFeasible);
    else
      return AssumeSymEQ(St, cast<lval::SymbolVal>(Cond).getSymbol(),
                         BasicVals.getZeroWithPtrWidth(), isFeasible);

  case lval::DeclValKind:
  case lval::FuncValKind:
  case lval::GotoLabelKind:
  case lval::StringLiteralValKind:
    isFeasible = Assumption;
    return St;

  case lval::FieldOffsetKind:
    return AssumeAux(St, cast<lval::FieldOffset>(Cond).getBase(),
                     Assumption, isFeasible);

  case lval::ArrayOffsetKind:
    return AssumeAux(St, cast<lval::ArrayOffset>(Cond).getBase(),
                     Assumption, isFeasible);

  case lval::ConcreteIntKind: {
    bool b = cast<lval::ConcreteInt>(Cond).getValue() != 0;
    isFeasible = b ? Assumption : !Assumption;
    return St;
  }
  } // end switch
}

const GRState*
BasicConstraintManager::Assume(const GRState* St, NonLVal Cond, bool Assumption,
                               bool& isFeasible) {
  St = AssumeAux(St, Cond, Assumption, isFeasible);
  // TF->EvalAssume() does nothing now.
  return St;
}

const GRState*
BasicConstraintManager::AssumeAux(const GRState* St,NonLVal Cond,
                                  bool Assumption, bool& isFeasible) {
  BasicValueFactory& BasicVals = StateMgr.getBasicVals();
  SymbolManager& SymMgr = StateMgr.getSymbolManager();

  switch (Cond.getSubKind()) {
  default:
    assert(false && "'Assume' not implemented for this NonLVal");

  case nonlval::SymbolValKind: {
    nonlval::SymbolVal& SV = cast<nonlval::SymbolVal>(Cond);
    SymbolID sym = SV.getSymbol();

    if (Assumption)
      return AssumeSymNE(St, sym, BasicVals.getValue(0, SymMgr.getType(sym)),
                         isFeasible);
    else
      return AssumeSymEQ(St, sym, BasicVals.getValue(0, SymMgr.getType(sym)),
                         isFeasible);
  }

  case nonlval::SymIntConstraintValKind:
    return
      AssumeSymInt(St, Assumption,
                   cast<nonlval::SymIntConstraintVal>(Cond).getConstraint(),
                   isFeasible);

  case nonlval::ConcreteIntKind: {
    bool b = cast<nonlval::ConcreteInt>(Cond).getValue() != 0;
    isFeasible = b ? Assumption : !Assumption;
    return St;
  }

  case nonlval::LValAsIntegerKind:
    return AssumeAux(St, cast<nonlval::LValAsInteger>(Cond).getLVal(),
                     Assumption, isFeasible);
  } // end switch
}

const GRState*
BasicConstraintManager::AssumeSymInt(const GRState* St, bool Assumption,
                                  const SymIntConstraint& C, bool& isFeasible) {

  switch (C.getOpcode()) {
  default:
    // No logic yet for other operators.
    isFeasible = true;
    return St;

  case BinaryOperator::EQ:
    if (Assumption)
      return AssumeSymEQ(St, C.getSymbol(), C.getInt(), isFeasible);
    else
      return AssumeSymNE(St, C.getSymbol(), C.getInt(), isFeasible);

  case BinaryOperator::NE:
    if (Assumption)
      return AssumeSymNE(St, C.getSymbol(), C.getInt(), isFeasible);
    else
      return AssumeSymEQ(St, C.getSymbol(), C.getInt(), isFeasible);

  case BinaryOperator::GE:
    if (Assumption)
      return AssumeSymGE(St, C.getSymbol(), C.getInt(), isFeasible);
    else
      return AssumeSymLT(St, C.getSymbol(), C.getInt(), isFeasible);

  case BinaryOperator::LE:
    if (Assumption)
      return AssumeSymLE(St, C.getSymbol(), C.getInt(), isFeasible);
    else
      return AssumeSymGT(St, C.getSymbol(), C.getInt(), isFeasible);
  } // end switch
}

const GRState*
BasicConstraintManager::AssumeSymNE(const GRState* St, SymbolID sym,
                                    const llvm::APSInt& V, bool& isFeasible) {
  // First, determine if sym == X, where X != V.
  if (const llvm::APSInt* X = St->getSymVal(sym)) {
    isFeasible = (*X != V);
    return St;
  }

  // Second, determine if sym != V.
  if (St->isNotEqual(sym, V)) {
    isFeasible = true;
    return St;
  }

  // If we reach here, sym is not a constant and we don't know if it is != V.
  // Make that assumption.
  isFeasible = true;
  return StateMgr.AddNE(St, sym, V);
}

const GRState*
BasicConstraintManager::AssumeSymEQ(const GRState* St, SymbolID sym,
                                    const llvm::APSInt& V, bool& isFeasible) {
  // First, determine if sym == X, where X != V.
  if (const llvm::APSInt* X = St->getSymVal(sym)) {
    isFeasible = *X == V;
    return St;
  }

  // Second, determine if sym != V.
  if (St->isNotEqual(sym, V)) {
    isFeasible = false;
    return St;
  }

  // If we reach here, sym is not a constant and we don't know if it is == V.
  // Make that assumption.

  isFeasible = true;
  return StateMgr.AddEQ(St, sym, V);
}

// These logic will be handled in another ConstraintManager.
const GRState*
BasicConstraintManager::AssumeSymLT(const GRState* St, SymbolID sym,
                                    const llvm::APSInt& V, bool& isFeasible) {

  // FIXME: For now have assuming x < y be the same as assuming sym != V;
  return AssumeSymNE(St, sym, V, isFeasible);
}

const GRState*
BasicConstraintManager::AssumeSymGT(const GRState* St, SymbolID sym,
                                    const llvm::APSInt& V, bool& isFeasible) {

  // FIXME: For now have assuming x > y be the same as assuming sym != V;
  return AssumeSymNE(St, sym, V, isFeasible);
}

const GRState*
BasicConstraintManager::AssumeSymGE(const GRState* St, SymbolID sym,
                                    const llvm::APSInt& V, bool& isFeasible) {

  // FIXME: Primitive logic for now.  Only reject a path if the value of
  //  sym is a constant X and !(X >= V).

  if (const llvm::APSInt* X = St->getSymVal(sym)) {
    isFeasible = *X >= V;
    return St;
  }

  isFeasible = true;
  return St;
}

const GRState*
BasicConstraintManager::AssumeSymLE(const GRState* St, SymbolID sym,
                                    const llvm::APSInt& V, bool& isFeasible) {

  // FIXME: Primitive logic for now.  Only reject a path if the value of
  //  sym is a constant X and !(X <= V).

  if (const llvm::APSInt* X = St->getSymVal(sym)) {
    isFeasible = *X <= V;
    return St;
  }

  isFeasible = true;
  return St;
}
