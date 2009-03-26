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

/// getAsLocSymbol - If this SVal is a location (subclasses Loc) and 
///  wraps a symbol, return that SymbolRef.  Otherwise return a SymbolRef
///  where 'isValid()' returns false.
SymbolRef SVal::getAsLocSymbol() const {
  if (const loc::SymbolVal *X = dyn_cast<loc::SymbolVal>(this))
    return X->getSymbol();

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
///  Otherwise return a SymbolRef where 'isValid()' returns false.
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

NonLoc Loc::EQ(SymbolManager& SymMgr, const Loc& R) const {
  
  switch (getSubKind()) {
    default:
      assert(false && "EQ not implemented for this Loc.");
      break;
      
    case loc::ConcreteIntKind:
      if (isa<loc::ConcreteInt>(R)) {
        bool b = cast<loc::ConcreteInt>(this)->getValue() ==
                 cast<loc::ConcreteInt>(R).getValue();
        
        return NonLoc::MakeIntTruthVal(SymMgr.getBasicVals(), b);
      }
      else if (isa<loc::SymbolVal>(R)) {
        const SymIntExpr *SE =
          SymMgr.getSymIntExpr(cast<loc::SymbolVal>(R).getSymbol(),
                               BinaryOperator::EQ,
                               cast<loc::ConcreteInt>(this)->getValue(),
                               SymMgr.getContext().IntTy);
        
        return nonloc::SymExprVal(SE);        
      }
      
      break;
      
      case loc::SymbolValKind: {
        if (isa<loc::ConcreteInt>(R)) {
          const SymIntExpr *SE =
            SymMgr.getSymIntExpr(cast<loc::SymbolVal>(this)->getSymbol(),
                                 BinaryOperator::EQ,
                                 cast<loc::ConcreteInt>(R).getValue(),
                                 SymMgr.getContext().IntTy);
          
          return nonloc::SymExprVal(SE);
        }
                                 
        assert (!isa<loc::SymbolVal>(R) && "FIXME: Implement unification.");        
        break;
      }
      
      case loc::MemRegionKind:
      if (isa<loc::MemRegionVal>(R)) {        
        bool b = cast<loc::MemRegionVal>(*this) == cast<loc::MemRegionVal>(R);
        return NonLoc::MakeIntTruthVal(SymMgr.getBasicVals(), b);
      }
      
      break;
  }
  
  return NonLoc::MakeIntTruthVal(SymMgr.getBasicVals(), false);
}

NonLoc Loc::NE(SymbolManager& SymMgr, const Loc& R) const {
  switch (getSubKind()) {
    default:
      assert(false && "NE not implemented for this Loc.");
      break;
      
    case loc::ConcreteIntKind:
      if (isa<loc::ConcreteInt>(R)) {
        bool b = cast<loc::ConcreteInt>(this)->getValue() !=
                 cast<loc::ConcreteInt>(R).getValue();
        
        return NonLoc::MakeIntTruthVal(SymMgr.getBasicVals(), b);
      }
      else if (isa<loc::SymbolVal>(R)) {
        const SymIntExpr *SE =
          SymMgr.getSymIntExpr(cast<loc::SymbolVal>(R).getSymbol(),
                               BinaryOperator::NE,
                               cast<loc::ConcreteInt>(this)->getValue(),
                               SymMgr.getContext().IntTy);
        return nonloc::SymExprVal(SE);
      }
      break;
      
      case loc::SymbolValKind: {
        if (isa<loc::ConcreteInt>(R)) {
          const SymIntExpr *SE =
            SymMgr.getSymIntExpr(cast<loc::SymbolVal>(this)->getSymbol(),
                                 BinaryOperator::NE,
                                 cast<loc::ConcreteInt>(R).getValue(),
                                 SymMgr.getContext().IntTy);
          
          return nonloc::SymExprVal(SE);
        }
        
        assert (!isa<loc::SymbolVal>(R) && "FIXME: Implement sym !=.");
        break;
      }
      
      case loc::MemRegionKind:
        if (isa<loc::MemRegionVal>(R)) {        
          bool b = cast<loc::MemRegionVal>(*this)==cast<loc::MemRegionVal>(R);
          return NonLoc::MakeIntTruthVal(SymMgr.getBasicVals(), b);
        }
      
        break;
  }
  
  return NonLoc::MakeIntTruthVal(SymMgr.getBasicVals(), true);
}

//===----------------------------------------------------------------------===//
// Utility methods for constructing Non-Locs.
//===----------------------------------------------------------------------===//

NonLoc NonLoc::MakeVal(SymbolRef sym) {
  return nonloc::SymbolVal(sym);
}

NonLoc NonLoc::MakeVal(SymbolManager& SymMgr, const SymExpr *lhs, 
                       BinaryOperator::Opcode op, const APSInt& v, QualType T) {
  // The Environment ensures we always get a persistent APSInt in
  // BasicValueFactory, so we don't need to get the APSInt from
  // BasicValueFactory again.
  assert(!Loc::IsLocType(T));
  return nonloc::SymExprVal(SymMgr.getSymIntExpr(lhs, op, v, T));
}

NonLoc NonLoc::MakeVal(SymbolManager& SymMgr, const SymExpr *lhs, 
                       BinaryOperator::Opcode op, const SymExpr *rhs,
QualType T) {
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

NonLoc NonLoc::MakeCompoundVal(QualType T, llvm::ImmutableList<SVal> Vals,
                               BasicValueFactory& BasicVals) {
  return nonloc::CompoundVal(BasicVals.getCompoundValData(T, Vals));
}

SVal SVal::GetRValueSymbolVal(SymbolManager& SymMgr, const MemRegion* R) {
  SymbolRef sym = SymMgr.getRegionRValueSymbol(R);
                                
  if (const TypedRegion* TR = dyn_cast<TypedRegion>(R)) {
    QualType T = TR->getRValueType(SymMgr.getContext());
    
    if (Loc::IsLocType(T))
      return Loc::MakeVal(sym);
  
    // Only handle integers for now.
    if (T->isIntegerType())
      return NonLoc::MakeVal(sym);
  }

  return UnknownVal();
}

SVal SVal::GetConjuredSymbolVal(SymbolManager &SymMgr, const Expr* E,
                                unsigned Count) {

  QualType T = E->getType();
  
  if (Loc::IsLocType(T)) {
    SymbolRef Sym = SymMgr.getConjuredSymbol(E, Count);        
    return loc::SymbolVal(Sym);
  }
  else if (T->isIntegerType() && T->isScalarType()) {
    SymbolRef Sym = SymMgr.getConjuredSymbol(E, Count);        
    return nonloc::SymbolVal(Sym);                    
  }

  return UnknownVal();
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

Loc Loc::MakeVal(SymbolRef sym) { return loc::SymbolVal(sym); }

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
      
    case loc::SymbolValKind:
      Out << '$' << cast<loc::SymbolVal>(this)->getSymbol();
      break;
      
    case loc::GotoLabelKind:
      Out << "&&"
          << cast<loc::GotoLabel>(this)->getLabel()->getID()->getName();
      break;

    case loc::MemRegionKind:
      Out << '&' << cast<loc::MemRegionVal>(this)->getRegion()->getString();
      break;
      
    case loc::FuncValKind:
      Out << "function " 
          << cast<loc::FuncVal>(this)->getDecl()->getIdentifier()->getName();
      break;
      
    default:
      assert (false && "Pretty-printing not implemented for this Loc.");
      break;
  }
}
