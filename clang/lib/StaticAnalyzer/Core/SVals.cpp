//= RValues.cpp - Abstract RValues for Path-Sens. Value Tracking -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SVal, Loc, and NonLoc, classes that represent
//  abstract r-values for use with path-sensitive value tracking.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/AST/ExprObjC.h"
#include "clang/Basic/IdentifierTable.h"
using namespace clang;
using namespace ento;
using llvm::APSInt;

//===----------------------------------------------------------------------===//
// Symbol iteration within an SVal.
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Utility methods.
//===----------------------------------------------------------------------===//

bool SVal::hasConjuredSymbol() const {
  if (const nonloc::SymbolVal* SV = dyn_cast<nonloc::SymbolVal>(this)) {
    SymbolRef sym = SV->getSymbol();
    if (isa<SymbolConjured>(sym))
      return true;
  }

  if (const loc::MemRegionVal *RV = dyn_cast<loc::MemRegionVal>(this)) {
    const MemRegion *R = RV->getRegion();
    if (const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(R)) {
      SymbolRef sym = SR->getSymbol();
      if (isa<SymbolConjured>(sym))
        return true;
    }
  }

  return false;
}

const FunctionDecl *SVal::getAsFunctionDecl() const {
  if (const loc::MemRegionVal* X = dyn_cast<loc::MemRegionVal>(this)) {
    const MemRegion* R = X->getRegion();
    if (const FunctionTextRegion *CTR = R->getAs<FunctionTextRegion>())
      return CTR->getDecl();
  }

  return NULL;
}

/// getAsLocSymbol - If this SVal is a location (subclasses Loc) and
///  wraps a symbol, return that SymbolRef.  Otherwise return 0.
// FIXME: should we consider SymbolRef wrapped in CodeTextRegion?
SymbolRef SVal::getAsLocSymbol() const {
  if (const nonloc::LocAsInteger *X = dyn_cast<nonloc::LocAsInteger>(this))
    return X->getLoc().getAsLocSymbol();

  if (const loc::MemRegionVal *X = dyn_cast<loc::MemRegionVal>(this)) {
    const MemRegion *R = X->stripCasts();
    if (const SymbolicRegion *SymR = dyn_cast<SymbolicRegion>(R))
      return SymR->getSymbol();
  }
  return NULL;
}

/// Get the symbol in the SVal or its base region.
SymbolRef SVal::getLocSymbolInBase() const {
  const loc::MemRegionVal *X = dyn_cast<loc::MemRegionVal>(this);

  if (!X)
    return 0;

  const MemRegion *R = X->getRegion();

  while (const SubRegion *SR = dyn_cast<SubRegion>(R)) {
    if (const SymbolicRegion *SymR = dyn_cast<SymbolicRegion>(SR))
      return SymR->getSymbol();
    else
      R = SR->getSuperRegion();
  }

  return 0;
}

/// getAsSymbol - If this Sval wraps a symbol return that SymbolRef.
///  Otherwise return 0.
// FIXME: should we consider SymbolRef wrapped in CodeTextRegion?
SymbolRef SVal::getAsSymbol() const {
  if (const nonloc::SymbolVal *X = dyn_cast<nonloc::SymbolVal>(this))
    return X->getSymbol();

  if (const nonloc::SymExprVal *X = dyn_cast<nonloc::SymExprVal>(this))
    if (SymbolRef Y = dyn_cast<SymbolData>(X->getSymbolicExpression()))
      return Y;

  return getAsLocSymbol();
}

/// getAsSymbolicExpression - If this Sval wraps a symbolic expression then
///  return that expression.  Otherwise return NULL.
const SymExpr *SVal::getAsSymbolicExpression() const {
  if (const nonloc::SymExprVal *X = dyn_cast<nonloc::SymExprVal>(this))
    return X->getSymbolicExpression();

  return getAsSymbol();
}

const MemRegion *SVal::getAsRegion() const {
  if (const loc::MemRegionVal *X = dyn_cast<loc::MemRegionVal>(this))
    return X->getRegion();

  if (const nonloc::LocAsInteger *X = dyn_cast<nonloc::LocAsInteger>(this)) {
    return X->getLoc().getAsRegion();
  }

  return 0;
}

const MemRegion *loc::MemRegionVal::stripCasts() const {
  const MemRegion *R = getRegion();
  return R ?  R->StripCasts() : NULL;
}

bool SVal::symbol_iterator::operator==(const symbol_iterator &X) const {
  return itr == X.itr;
}

bool SVal::symbol_iterator::operator!=(const symbol_iterator &X) const {
  return itr != X.itr;
}

SVal::symbol_iterator::symbol_iterator(const SymExpr *SE) {
  itr.push_back(SE);
  while (!isa<SymbolData>(itr.back())) expand();
}

SVal::symbol_iterator &SVal::symbol_iterator::operator++() {
  assert(!itr.empty() && "attempting to iterate on an 'end' iterator");
  assert(isa<SymbolData>(itr.back()));
  itr.pop_back();
  if (!itr.empty())
    while (!isa<SymbolData>(itr.back())) expand();
  return *this;
}

SymbolRef SVal::symbol_iterator::operator*() {
  assert(!itr.empty() && "attempting to dereference an 'end' iterator");
  return cast<SymbolData>(itr.back());
}

void SVal::symbol_iterator::expand() {
  const SymExpr *SE = itr.back();
  itr.pop_back();

  if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(SE)) {
    itr.push_back(SIE->getLHS());
    return;
  }
  else if (const SymSymExpr *SSE = dyn_cast<SymSymExpr>(SE)) {
    itr.push_back(SSE->getLHS());
    itr.push_back(SSE->getRHS());
    return;
  }

  assert(false && "unhandled expansion case");
}

const void *nonloc::LazyCompoundVal::getStore() const {
  return static_cast<const LazyCompoundValData*>(Data)->getStore();
}

const TypedRegion *nonloc::LazyCompoundVal::getRegion() const {
  return static_cast<const LazyCompoundValData*>(Data)->getRegion();
}

//===----------------------------------------------------------------------===//
// Other Iterators.
//===----------------------------------------------------------------------===//

nonloc::CompoundVal::iterator nonloc::CompoundVal::begin() const {
  return getValue()->begin();
}

nonloc::CompoundVal::iterator nonloc::CompoundVal::end() const {
  return getValue()->end();
}

//===----------------------------------------------------------------------===//
// Useful predicates.
//===----------------------------------------------------------------------===//

bool SVal::isConstant() const {
  return isa<nonloc::ConcreteInt>(this) || isa<loc::ConcreteInt>(this);
}

bool SVal::isConstant(int I) const {
  if (isa<loc::ConcreteInt>(*this))
    return cast<loc::ConcreteInt>(*this).getValue() == I;
  else if (isa<nonloc::ConcreteInt>(*this))
    return cast<nonloc::ConcreteInt>(*this).getValue() == I;
  else
    return false;
}

bool SVal::isZeroConstant() const {
  return isConstant(0);
}


//===----------------------------------------------------------------------===//
// Transfer function dispatch for Non-Locs.
//===----------------------------------------------------------------------===//

SVal nonloc::ConcreteInt::evalBinOp(SValBuilder &svalBuilder,
                                    BinaryOperator::Opcode Op,
                                    const nonloc::ConcreteInt& R) const {
  const llvm::APSInt* X =
    svalBuilder.getBasicValueFactory().evalAPSInt(Op, getValue(), R.getValue());

  if (X)
    return nonloc::ConcreteInt(*X);
  else
    return UndefinedVal();
}

nonloc::ConcreteInt
nonloc::ConcreteInt::evalComplement(SValBuilder &svalBuilder) const {
  return svalBuilder.makeIntVal(~getValue());
}

nonloc::ConcreteInt
nonloc::ConcreteInt::evalMinus(SValBuilder &svalBuilder) const {
  return svalBuilder.makeIntVal(-getValue());
}

//===----------------------------------------------------------------------===//
// Transfer function dispatch for Locs.
//===----------------------------------------------------------------------===//

SVal loc::ConcreteInt::evalBinOp(BasicValueFactory& BasicVals,
                                 BinaryOperator::Opcode Op,
                                 const loc::ConcreteInt& R) const {

  assert (Op == BO_Add || Op == BO_Sub ||
          (Op >= BO_LT && Op <= BO_NE));

  const llvm::APSInt* X = BasicVals.evalAPSInt(Op, getValue(), R.getValue());

  if (X)
    return loc::ConcreteInt(*X);
  else
    return UndefinedVal();
}

//===----------------------------------------------------------------------===//
// Pretty-Printing.
//===----------------------------------------------------------------------===//

void SVal::dump() const { dumpToStream(llvm::errs()); }

void SVal::dumpToStream(raw_ostream &os) const {
  switch (getBaseKind()) {
    case UnknownKind:
      os << "Unknown";
      break;
    case NonLocKind:
      cast<NonLoc>(this)->dumpToStream(os);
      break;
    case LocKind:
      cast<Loc>(this)->dumpToStream(os);
      break;
    case UndefinedKind:
      os << "Undefined";
      break;
    default:
      assert (false && "Invalid SVal.");
  }
}

void NonLoc::dumpToStream(raw_ostream &os) const {
  switch (getSubKind()) {
    case nonloc::ConcreteIntKind: {
      const nonloc::ConcreteInt& C = *cast<nonloc::ConcreteInt>(this);
      if (C.getValue().isUnsigned())
        os << C.getValue().getZExtValue();
      else
        os << C.getValue().getSExtValue();
      os << ' ' << (C.getValue().isUnsigned() ? 'U' : 'S')
         << C.getValue().getBitWidth() << 'b';
      break;
    }
    case nonloc::SymbolValKind:
      os << '$' << cast<nonloc::SymbolVal>(this)->getSymbol();
      break;
    case nonloc::SymExprValKind: {
      const nonloc::SymExprVal& C = *cast<nonloc::SymExprVal>(this);
      const SymExpr *SE = C.getSymbolicExpression();
      os << SE;
      break;
    }
    case nonloc::LocAsIntegerKind: {
      const nonloc::LocAsInteger& C = *cast<nonloc::LocAsInteger>(this);
      os << C.getLoc() << " [as " << C.getNumBits() << " bit integer]";
      break;
    }
    case nonloc::CompoundValKind: {
      const nonloc::CompoundVal& C = *cast<nonloc::CompoundVal>(this);
      os << "compoundVal{";
      bool first = true;
      for (nonloc::CompoundVal::iterator I=C.begin(), E=C.end(); I!=E; ++I) {
        if (first) {
          os << ' '; first = false;
        }
        else
          os << ", ";

        (*I).dumpToStream(os);
      }
      os << "}";
      break;
    }
    case nonloc::LazyCompoundValKind: {
      const nonloc::LazyCompoundVal &C = *cast<nonloc::LazyCompoundVal>(this);
      os << "lazyCompoundVal{" << const_cast<void *>(C.getStore())
         << ',' << C.getRegion()
         << '}';
      break;
    }
    default:
      assert (false && "Pretty-printed not implemented for this NonLoc.");
      break;
  }
}

void Loc::dumpToStream(raw_ostream &os) const {
  switch (getSubKind()) {
    case loc::ConcreteIntKind:
      os << cast<loc::ConcreteInt>(this)->getValue().getZExtValue() << " (Loc)";
      break;
    case loc::GotoLabelKind:
      os << "&&" << cast<loc::GotoLabel>(this)->getLabel()->getName();
      break;
    case loc::MemRegionKind:
      os << '&' << cast<loc::MemRegionVal>(this)->getRegion()->getString();
      break;
    case loc::ObjCPropRefKind: {
      const ObjCPropertyRefExpr *E = cast<loc::ObjCPropRef>(this)->getPropRefExpr();
      os << "objc-prop{";
      if (E->isSuperReceiver())
        os << "super.";
      else if (E->getBase())
        os << "<base>.";

      if (E->isImplicitProperty())
        os << E->getImplicitPropertyGetter()->getSelector().getAsString();
      else
        os << E->getExplicitProperty()->getName();

      os << "}";
      break;
    }
    default:
      assert(false && "Pretty-printing not implemented for this Loc.");
      break;
  }
}
