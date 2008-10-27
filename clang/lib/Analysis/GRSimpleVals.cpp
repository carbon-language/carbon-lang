// GRSimpleVals.cpp - Transfer functions for tracking simple values -*- C++ -*--
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines GRSimpleVals, a sub-class of GRTransferFuncs that
//  provides transfer functions for performing simple value tracking with
//  limited support for symbolics.
//
//===----------------------------------------------------------------------===//

#include "GRSimpleVals.h"
#include "BasicObjCFoundationChecks.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "llvm/Support/Compiler.h"
#include <sstream>

using namespace clang;

//===----------------------------------------------------------------------===//
// Transfer Function creation for External clients.
//===----------------------------------------------------------------------===//

GRTransferFuncs* clang::MakeGRSimpleValsTF() { return new GRSimpleVals(); }  

//===----------------------------------------------------------------------===//
// Transfer function for Casts.
//===----------------------------------------------------------------------===//

SVal GRSimpleVals::EvalCast(GRExprEngine& Eng, NonLoc X, QualType T) {
  
  if (!isa<nonloc::ConcreteInt>(X))
    return UnknownVal();

  bool isLocType = Loc::IsLocType(T);
  
  // Only handle casts from integers to integers.
  if (!isLocType && !T->isIntegerType())
    return UnknownVal();
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();
  
  llvm::APSInt V = cast<nonloc::ConcreteInt>(X).getValue();
  V.setIsUnsigned(T->isUnsignedIntegerType() || Loc::IsLocType(T));
  V.extOrTrunc(Eng.getContext().getTypeSize(T));
  
  if (isLocType)
    return loc::ConcreteInt(BasicVals.getValue(V));
  else
    return nonloc::ConcreteInt(BasicVals.getValue(V));
}

// Casts.

SVal GRSimpleVals::EvalCast(GRExprEngine& Eng, Loc X, QualType T) {
  
  // Casts from pointers -> pointers, just return the lval.
  //
  // Casts from pointers -> references, just return the lval.  These
  //   can be introduced by the frontend for corner cases, e.g
  //   casting from va_list* to __builtin_va_list&.
  //
  if (Loc::IsLocType(T) || T->isReferenceType())
    return X;
  
  assert (T->isIntegerType());
  
  if (!isa<loc::ConcreteInt>(X))
    return UnknownVal();
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();
  
  llvm::APSInt V = cast<loc::ConcreteInt>(X).getValue();
  V.setIsUnsigned(T->isUnsignedIntegerType() || Loc::IsLocType(T));
  V.extOrTrunc(Eng.getContext().getTypeSize(T));

  return nonloc::ConcreteInt(BasicVals.getValue(V));
}

// Unary operators.

SVal GRSimpleVals::EvalMinus(GRExprEngine& Eng, UnaryOperator* U, NonLoc X){
  
  switch (X.getSubKind()) {
      
    case nonloc::ConcreteIntKind:
      return cast<nonloc::ConcreteInt>(X).EvalMinus(Eng.getBasicVals(), U);
      
    default:
      return UnknownVal();
  }
}

SVal GRSimpleVals::EvalComplement(GRExprEngine& Eng, NonLoc X) {

  switch (X.getSubKind()) {
      
    case nonloc::ConcreteIntKind:
      return cast<nonloc::ConcreteInt>(X).EvalComplement(Eng.getBasicVals());
      
    default:
      return UnknownVal();
  }
}

// Binary operators.

static unsigned char LNotOpMap[] = {
  (unsigned char) BinaryOperator::GE,  /* LT => GE */
  (unsigned char) BinaryOperator::LE,  /* GT => LE */
  (unsigned char) BinaryOperator::GT,  /* LE => GT */
  (unsigned char) BinaryOperator::LT,  /* GE => LT */
  (unsigned char) BinaryOperator::NE,  /* EQ => NE */
  (unsigned char) BinaryOperator::EQ   /* NE => EQ */
};

SVal GRSimpleVals::DetermEvalBinOpNN(GRStateManager& StateMgr,
                                     BinaryOperator::Opcode Op,
                                     NonLoc L, NonLoc R)  {

  BasicValueFactory& BasicVals = StateMgr.getBasicVals();
  unsigned subkind = L.getSubKind();
  
  while (1) {
    
    switch (subkind) {
      default:
        return UnknownVal();
        
      case nonloc::SymIntConstraintValKind: {
        
        // Logical not?        
        if (!(Op == BinaryOperator::EQ && R.isZeroConstant()))
          return UnknownVal();
        
        const SymIntConstraint& C =
          cast<nonloc::SymIntConstraintVal>(L).getConstraint();
        
        BinaryOperator::Opcode Opc = C.getOpcode();
        
        if (Opc < BinaryOperator::LT || Opc > BinaryOperator::NE)
          return UnknownVal();

        // For comparison operators, translate the constraint by
        // changing the opcode.
        
        int idx = (unsigned) Opc - (unsigned) BinaryOperator::LT;
        
        assert (idx >= 0 && 
                (unsigned) idx < sizeof(LNotOpMap)/sizeof(unsigned char));
        
        Opc = (BinaryOperator::Opcode) LNotOpMap[idx];
        
        const SymIntConstraint& CNew =
          BasicVals.getConstraint(C.getSymbol(), Opc, C.getInt());
        
        return nonloc::SymIntConstraintVal(CNew);
      }
        
      case nonloc::ConcreteIntKind:
        
        if (isa<nonloc::ConcreteInt>(R)) {          
          const nonloc::ConcreteInt& L_CI = cast<nonloc::ConcreteInt>(L);
          const nonloc::ConcreteInt& R_CI = cast<nonloc::ConcreteInt>(R);
          return L_CI.EvalBinOp(BasicVals, Op, R_CI);          
        }
        else {
          subkind = R.getSubKind();
          NonLoc tmp = R;
          R = L;
          L = tmp;
          
          // Swap the operators.
          switch (Op) {
            case BinaryOperator::LT: Op = BinaryOperator::GT; break;
            case BinaryOperator::GT: Op = BinaryOperator::LT; break;
            case BinaryOperator::LE: Op = BinaryOperator::GE; break;
            case BinaryOperator::GE: Op = BinaryOperator::LE; break;
            default: break;
          }
          
          continue;
        }
        
      case nonloc::SymbolValKind:
        if (isa<nonloc::ConcreteInt>(R)) {
          const SymIntConstraint& C =
            BasicVals.getConstraint(cast<nonloc::SymbolVal>(L).getSymbol(), Op,
                                    cast<nonloc::ConcreteInt>(R).getValue());
          
          return nonloc::SymIntConstraintVal(C);
        }
        else
          return UnknownVal();
    }
  }
}


// Binary Operators (except assignments and comma).

SVal GRSimpleVals::EvalBinOp(GRExprEngine& Eng, BinaryOperator::Opcode Op,
                             Loc L, Loc R) {
  
  switch (Op) {

    default:
      return UnknownVal();
      
    case BinaryOperator::EQ:
      return EvalEQ(Eng, L, R);
      
    case BinaryOperator::NE:
      return EvalNE(Eng, L, R);      
  }
}

// Pointer arithmetic.

SVal GRSimpleVals::EvalBinOp(GRExprEngine& Eng, BinaryOperator::Opcode Op,
                             Loc L, NonLoc R) {  
  return UnknownVal();
}

// Equality operators for Locs.

SVal GRSimpleVals::EvalEQ(GRExprEngine& Eng, Loc L, Loc R) {
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();
  
  switch (L.getSubKind()) {

    default:
      assert(false && "EQ not implemented for this Loc.");
      return UnknownVal();
      
    case loc::ConcreteIntKind:

      if (isa<loc::ConcreteInt>(R)) {
        bool b = cast<loc::ConcreteInt>(L).getValue() ==
                 cast<loc::ConcreteInt>(R).getValue();
        
        return NonLoc::MakeIntTruthVal(BasicVals, b);
      }
      else if (isa<loc::SymbolVal>(R)) {
        
        const SymIntConstraint& C =
          BasicVals.getConstraint(cast<loc::SymbolVal>(R).getSymbol(),
                               BinaryOperator::EQ,
                               cast<loc::ConcreteInt>(L).getValue());
        
        return nonloc::SymIntConstraintVal(C);
      }
      
      break;
      
    case loc::SymbolValKind: {

      if (isa<loc::ConcreteInt>(R)) {          
        const SymIntConstraint& C =
          BasicVals.getConstraint(cast<loc::SymbolVal>(L).getSymbol(),
                               BinaryOperator::EQ,
                               cast<loc::ConcreteInt>(R).getValue());
        
        return nonloc::SymIntConstraintVal(C);
      }
      
      // FIXME: Implement == for lval Symbols.  This is mainly useful
      //  in iterator loops when traversing a buffer, e.g. while(z != zTerm).
      //  Since this is not useful for many checkers we'll punt on this for 
      //  now.
       
      return UnknownVal();      
    }
      
    case loc::MemRegionKind:
    case loc::FuncValKind:
    case loc::GotoLabelKind:
      return NonLoc::MakeIntTruthVal(BasicVals, L == R);
  }
  
  return NonLoc::MakeIntTruthVal(BasicVals, false);
}

SVal GRSimpleVals::EvalNE(GRExprEngine& Eng, Loc L, Loc R) {
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();

  switch (L.getSubKind()) {

    default:
      assert(false && "NE not implemented for this Loc.");
      return UnknownVal();
      
    case loc::ConcreteIntKind:
      
      if (isa<loc::ConcreteInt>(R)) {
        bool b = cast<loc::ConcreteInt>(L).getValue() !=
                 cast<loc::ConcreteInt>(R).getValue();
        
        return NonLoc::MakeIntTruthVal(BasicVals, b);
      }
      else if (isa<loc::SymbolVal>(R)) {        
        const SymIntConstraint& C =
          BasicVals.getConstraint(cast<loc::SymbolVal>(R).getSymbol(),
                                  BinaryOperator::NE,
                                  cast<loc::ConcreteInt>(L).getValue());
        
        return nonloc::SymIntConstraintVal(C);
      }
      
      break;
      
    case loc::SymbolValKind: {
      if (isa<loc::ConcreteInt>(R)) {          
        const SymIntConstraint& C =
          BasicVals.getConstraint(cast<loc::SymbolVal>(L).getSymbol(),
                                  BinaryOperator::NE,
                                  cast<loc::ConcreteInt>(R).getValue());
        
        return nonloc::SymIntConstraintVal(C);
      }
      
      // FIXME: Implement != for lval Symbols.  This is mainly useful
      //  in iterator loops when traversing a buffer, e.g. while(z != zTerm).
      //  Since this is not useful for many checkers we'll punt on this for 
      //  now.
      
      return UnknownVal();
      
      break;
    }
      
    case loc::MemRegionKind:
    case loc::FuncValKind:
    case loc::GotoLabelKind:
      return NonLoc::MakeIntTruthVal(BasicVals, L != R);
  }
  
  return NonLoc::MakeIntTruthVal(BasicVals, true);
}

//===----------------------------------------------------------------------===//
// Transfer function for function calls.
//===----------------------------------------------------------------------===//

void GRSimpleVals::EvalCall(ExplodedNodeSet<GRState>& Dst,
                            GRExprEngine& Eng,
                            GRStmtNodeBuilder<GRState>& Builder,
                            CallExpr* CE, SVal L,
                            ExplodedNode<GRState>* Pred) {
  
  GRStateManager& StateMgr = Eng.getStateManager();
  const GRState* St = Builder.GetState(Pred);
  
  // Invalidate all arguments passed in by reference (Locs).

  for (CallExpr::arg_iterator I = CE->arg_begin(), E = CE->arg_end();
        I != E; ++I) {

    SVal V = StateMgr.GetSVal(St, *I);
    
    if (isa<loc::MemRegionVal>(V))
      St = StateMgr.SetSVal(St, cast<Loc>(V), UnknownVal());
    else if (isa<nonloc::LocAsInteger>(V))
      St = StateMgr.SetSVal(St, cast<nonloc::LocAsInteger>(V).getLoc(),
                            UnknownVal());
    
  }
  
  // Make up a symbol for the return value of this function.  
  // FIXME: We eventually should handle structs and other compound types
  // that are returned by value.
  QualType T = CE->getType();  
  if (T->isIntegerType() || Loc::IsLocType(T)) {    
    unsigned Count = Builder.getCurrentBlockCount();
    SymbolID Sym = Eng.getSymbolManager().getConjuredSymbol(CE, Count);
        
    SVal X = Loc::IsLocType(CE->getType())
             ? cast<SVal>(loc::SymbolVal(Sym)) 
             : cast<SVal>(nonloc::SymbolVal(Sym));
    
    St = StateMgr.SetSVal(St, CE, X, Eng.getCFG().isBlkExpr(CE), false);
  }  
    
  Builder.MakeNode(Dst, CE, Pred, St);
}

//===----------------------------------------------------------------------===//
// Transfer function for Objective-C message expressions.
//===----------------------------------------------------------------------===//

void GRSimpleVals::EvalObjCMessageExpr(ExplodedNodeSet<GRState>& Dst,
                                       GRExprEngine& Eng,
                                       GRStmtNodeBuilder<GRState>& Builder,
                                       ObjCMessageExpr* ME,
                                       ExplodedNode<GRState>* Pred) {
  
  
  // The basic transfer function logic for message expressions does nothing.
  // We just invalidate all arguments passed in by references.
  
  GRStateManager& StateMgr = Eng.getStateManager();
  const GRState* St = Builder.GetState(Pred);
  
  for (ObjCMessageExpr::arg_iterator I = ME->arg_begin(), E = ME->arg_end();
       I != E; ++I) {
    
    SVal V = StateMgr.GetSVal(St, *I);
    
    if (isa<Loc>(V))
      St = StateMgr.SetSVal(St, cast<Loc>(V), UnknownVal());
  }
  
  Builder.MakeNode(Dst, ME, Pred, St);
}
