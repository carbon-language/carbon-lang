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

#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/Support/Streams.h"

using namespace clang;
using llvm::dyn_cast;
using llvm::cast;
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
    } else if (const CodeTextRegion *CTR = dyn_cast<CodeTextRegion>(R)) {
      if (CTR->isSymbolic()) {
        SymbolRef sym = CTR->getSymbol();
        if (isa<SymbolConjured>(sym))
          return true;
      }
    }
  }

  return false;
}

const FunctionDecl* SVal::getAsFunctionDecl() const {
  if (const loc::MemRegionVal* X = dyn_cast<loc::MemRegionVal>(this)) {
    const MemRegion* R = X->getRegion();
    if (const CodeTextRegion* CTR = R->getAs<CodeTextRegion>()) {
      if (CTR->isDeclared())
        return CTR->getDecl();
    }
  }

  return 0;
}

/// getAsLocSymbol - If this SVal is a location (subclasses Loc) and 
///  wraps a symbol, return that SymbolRef.  Otherwise return 0.
// FIXME: should we consider SymbolRef wrapped in CodeTextRegion?
SymbolRef SVal::getAsLocSymbol() const {
  if (const loc::MemRegionVal *X = dyn_cast<loc::MemRegionVal>(this)) {
    const MemRegion *R = X->getRegion();
    
    while (R) {
      // Blast through region views.
      if (const TypedViewRegion *View = dyn_cast<TypedViewRegion>(R)) {
        R = View->getSuperRegion();
        continue;
      }
      
      if (const SymbolicRegion *SymR = dyn_cast<SymbolicRegion>(R))
        return SymR->getSymbol();
      
      break;
    }
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

SVal::symbol_iterator& SVal::symbol_iterator::operator++() {
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

bool SVal::isZeroConstant() const {
  if (isa<loc::ConcreteInt>(*this))
    return cast<loc::ConcreteInt>(*this).getValue() == 0;
  else if (isa<nonloc::ConcreteInt>(*this))
    return cast<nonloc::ConcreteInt>(*this).getValue() == 0;
  else
    return false;
}


//===----------------------------------------------------------------------===//
// Transfer function dispatch for Non-Locs.
//===----------------------------------------------------------------------===//

SVal nonloc::ConcreteInt::EvalBinOp(BasicValueFactory& BasicVals,
                                     BinaryOperator::Opcode Op,
                                     const nonloc::ConcreteInt& R) const {
  
  const llvm::APSInt* X =
    BasicVals.EvaluateAPSInt(Op, getValue(), R.getValue());
  
  if (X)
    return nonloc::ConcreteInt(*X);
  else
    return UndefinedVal();
}

  // Bitwise-Complement.

nonloc::ConcreteInt
nonloc::ConcreteInt::EvalComplement(BasicValueFactory& BasicVals) const {
  return BasicVals.getValue(~getValue()); 
}

  // Unary Minus.

nonloc::ConcreteInt
nonloc::ConcreteInt::EvalMinus(BasicValueFactory& BasicVals, UnaryOperator* U) const {
  assert (U->getType() == U->getSubExpr()->getType());  
  assert (U->getType()->isIntegerType());  
  return BasicVals.getValue(-getValue()); 
}

//===----------------------------------------------------------------------===//
// Transfer function dispatch for Locs.
//===----------------------------------------------------------------------===//

SVal loc::ConcreteInt::EvalBinOp(BasicValueFactory& BasicVals,
                                 BinaryOperator::Opcode Op,
                                 const loc::ConcreteInt& R) const {
  
  assert (Op == BinaryOperator::Add || Op == BinaryOperator::Sub ||
          (Op >= BinaryOperator::LT && Op <= BinaryOperator::NE));
  
  const llvm::APSInt* X = BasicVals.EvaluateAPSInt(Op, getValue(), R.getValue());
  
  if (X)
    return loc::ConcreteInt(*X);
  else
    return UndefinedVal();
}

//===----------------------------------------------------------------------===//
// Utility methods for constructing SVals.
//===----------------------------------------------------------------------===//

SVal ValueManager::makeZeroVal(QualType T) {
  if (Loc::IsLocType(T))
    return Loc::MakeNull(BasicVals);

  if (T->isIntegerType())
    return NonLoc::MakeVal(BasicVals, 0, T);
  
  // FIXME: Handle floats.
  // FIXME: Handle structs.
  return UnknownVal();  
}

SVal ValueManager::makeZeroArrayIndex() {
  return nonloc::ConcreteInt(BasicVals.getZeroWithPtrWidth(false));
}

//===----------------------------------------------------------------------===//
// Utility methods for constructing Non-Locs.
//===----------------------------------------------------------------------===//

NonLoc ValueManager::makeNonLoc(SymbolRef sym) {
  return nonloc::SymbolVal(sym);
}

NonLoc ValueManager::makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                                const APSInt& v, QualType T) {
  // The Environment ensures we always get a persistent APSInt in
  // BasicValueFactory, so we don't need to get the APSInt from
  // BasicValueFactory again.
  assert(!Loc::IsLocType(T));
  return nonloc::SymExprVal(SymMgr.getSymIntExpr(lhs, op, v, T));
}

NonLoc ValueManager::makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                                const SymExpr *rhs, QualType T) {
  assert(SymMgr.getType(lhs) == SymMgr.getType(rhs));
  assert(!Loc::IsLocType(T));
  return nonloc::SymExprVal(SymMgr.getSymSymExpr(lhs, op, rhs, T));
}

NonLoc NonLoc::MakeIntVal(BasicValueFactory& BasicVals, uint64_t X, 
                          bool isUnsigned) {
  return nonloc::ConcreteInt(BasicVals.getIntValue(X, isUnsigned));
}

NonLoc NonLoc::MakeVal(BasicValueFactory& BasicVals, uint64_t X, 
                       unsigned BitWidth, bool isUnsigned) {
  return nonloc::ConcreteInt(BasicVals.getValue(X, BitWidth, isUnsigned));
}

NonLoc NonLoc::MakeVal(BasicValueFactory& BasicVals, uint64_t X, QualType T) {  
  return nonloc::ConcreteInt(BasicVals.getValue(X, T));
}

NonLoc NonLoc::MakeVal(BasicValueFactory& BasicVals, IntegerLiteral* I) {

  return nonloc::ConcreteInt(BasicVals.getValue(APSInt(I->getValue(),
                              I->getType()->isUnsignedIntegerType())));
}

NonLoc NonLoc::MakeVal(BasicValueFactory& BasicVals, const llvm::APInt& I,
                       bool isUnsigned) {
  return nonloc::ConcreteInt(BasicVals.getValue(I, isUnsigned));
}

NonLoc NonLoc::MakeVal(BasicValueFactory& BasicVals, const llvm::APSInt& I) {
  return nonloc::ConcreteInt(BasicVals.getValue(I));
}

NonLoc NonLoc::MakeIntTruthVal(BasicValueFactory& BasicVals, bool b) {
  return nonloc::ConcreteInt(BasicVals.getTruthValue(b));
}

NonLoc ValueManager::makeTruthVal(bool b, QualType T) {
  return nonloc::ConcreteInt(BasicVals.getTruthValue(b, T));
}

NonLoc NonLoc::MakeCompoundVal(QualType T, llvm::ImmutableList<SVal> Vals,
                               BasicValueFactory& BasicVals) {
  return nonloc::CompoundVal(BasicVals.getCompoundValData(T, Vals));
}

SVal ValueManager::getRegionValueSymbolVal(const MemRegion* R) {
  SymbolRef sym = SymMgr.getRegionValueSymbol(R);
                                
  if (const TypedRegion* TR = dyn_cast<TypedRegion>(R)) {
    QualType T = TR->getValueType(SymMgr.getContext());

    // If T is of function pointer type, create a CodeTextRegion wrapping a
    // symbol.
    if (T->isFunctionPointerType()) {
      return Loc::MakeVal(MemMgr.getCodeTextRegion(sym, T));
    }
    
    if (Loc::IsLocType(T))
      return Loc::MakeVal(MemMgr.getSymbolicRegion(sym));
  
    // Only handle integers for now.
    if (T->isIntegerType() && T->isScalarType())
      return makeNonLoc(sym);
  }

  return UnknownVal();
}

SVal ValueManager::getConjuredSymbolVal(const Expr* E, unsigned Count) {
  QualType T = E->getType();
  SymbolRef sym = SymMgr.getConjuredSymbol(E, Count);

  // If T is of function pointer type, create a CodeTextRegion wrapping a
  // symbol.
  if (T->isFunctionPointerType()) {
    return Loc::MakeVal(MemMgr.getCodeTextRegion(sym, T));
  }

  if (Loc::IsLocType(T))
    return Loc::MakeVal(MemMgr.getSymbolicRegion(sym));

  if (T->isIntegerType() && T->isScalarType())
    return makeNonLoc(sym);

  return UnknownVal();
}

SVal ValueManager::getConjuredSymbolVal(const Expr* E, QualType T,
                                        unsigned Count) {

  SymbolRef sym = SymMgr.getConjuredSymbol(E, T, Count);

  // If T is of function pointer type, create a CodeTextRegion wrapping a
  // symbol.
  if (T->isFunctionPointerType()) {
    return Loc::MakeVal(MemMgr.getCodeTextRegion(sym, T));
  }

  if (Loc::IsLocType(T))
    return Loc::MakeVal(MemMgr.getSymbolicRegion(sym));

  if (T->isIntegerType() && T->isScalarType())
    return makeNonLoc(sym);

  return UnknownVal();
}

SVal ValueManager::getFunctionPointer(const FunctionDecl* FD) {
  CodeTextRegion* R 
    = MemMgr.getCodeTextRegion(FD, Context.getPointerType(FD->getType()));
  return loc::MemRegionVal(R);
}

nonloc::LocAsInteger nonloc::LocAsInteger::Make(BasicValueFactory& Vals, Loc V,
                                                unsigned Bits) {
  return LocAsInteger(Vals.getPersistentSValWithData(V, Bits));
}

//===----------------------------------------------------------------------===//
// Utility methods for constructing Locs.
//===----------------------------------------------------------------------===//

Loc Loc::MakeVal(const MemRegion* R) { return loc::MemRegionVal(R); }

Loc Loc::MakeVal(AddrLabelExpr* E) { return loc::GotoLabel(E->getLabel()); }

Loc Loc::MakeNull(BasicValueFactory &BasicVals) {
  return loc::ConcreteInt(BasicVals.getZeroWithPtrWidth());
}

//===----------------------------------------------------------------------===//
// Pretty-Printing.
//===----------------------------------------------------------------------===//

void SVal::printStdErr() const { print(llvm::errs()); }

void SVal::print(std::ostream& Out) const {
  llvm::raw_os_ostream out(Out);
  print(out);
}

void SVal::print(llvm::raw_ostream& Out) const {

  switch (getBaseKind()) {
      
    case UnknownKind:
      Out << "Invalid"; break;
      
    case NonLocKind:
      cast<NonLoc>(this)->print(Out); break;
      
    case LocKind:
      cast<Loc>(this)->print(Out); break;
      
    case UndefinedKind:
      Out << "Undefined"; break;
      
    default:
      assert (false && "Invalid SVal.");
  }
}

void NonLoc::print(llvm::raw_ostream& Out) const {

  switch (getSubKind()) {  

    case nonloc::ConcreteIntKind:
      Out << cast<nonloc::ConcreteInt>(this)->getValue().getZExtValue();

      if (cast<nonloc::ConcreteInt>(this)->getValue().isUnsigned())
        Out << 'U';
      
      break;
      
    case nonloc::SymbolValKind:
      Out << '$' << cast<nonloc::SymbolVal>(this)->getSymbol();
      break;
     
    case nonloc::SymExprValKind: {
      const nonloc::SymExprVal& C = *cast<nonloc::SymExprVal>(this);
      const SymExpr *SE = C.getSymbolicExpression();
      Out << SE;
      break;
    }
    
    case nonloc::LocAsIntegerKind: {
      const nonloc::LocAsInteger& C = *cast<nonloc::LocAsInteger>(this);
      C.getLoc().print(Out);
      Out << " [as " << C.getNumBits() << " bit integer]";
      break;
    }
      
    case nonloc::CompoundValKind: {
      const nonloc::CompoundVal& C = *cast<nonloc::CompoundVal>(this);
      Out << " {";
      bool first = true;
      for (nonloc::CompoundVal::iterator I=C.begin(), E=C.end(); I!=E; ++I) {
        if (first) { Out << ' '; first = false; }
        else Out << ", ";
        (*I).print(Out);
      }
      Out << " }";
      break;
    }
      
    default:
      assert (false && "Pretty-printed not implemented for this NonLoc.");
      break;
  }
}

void Loc::print(llvm::raw_ostream& Out) const {
  
  switch (getSubKind()) {        

    case loc::ConcreteIntKind:
      Out << cast<loc::ConcreteInt>(this)->getValue().getZExtValue()
          << " (Loc)";
      break;
      
    case loc::GotoLabelKind:
      Out << "&&"
          << cast<loc::GotoLabel>(this)->getLabel()->getID()->getName();
      break;

    case loc::MemRegionKind:
      Out << '&' << cast<loc::MemRegionVal>(this)->getRegion()->getString();
      break;
      
    default:
      assert (false && "Pretty-printing not implemented for this Loc.");
      break;
  }
}
