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

RVal GRSimpleVals::EvalCast(GRExprEngine& Eng, NonLVal X, QualType T) {
  
  if (!isa<nonlval::ConcreteInt>(X))
    return UnknownVal();

  bool isLValType = LVal::IsLValType(T);
  
  // Only handle casts from integers to integers.
  if (!isLValType && !T->isIntegerType())
    return UnknownVal();
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();
  
  llvm::APSInt V = cast<nonlval::ConcreteInt>(X).getValue();
  V.setIsUnsigned(T->isUnsignedIntegerType() || LVal::IsLValType(T));
  V.extOrTrunc(Eng.getContext().getTypeSize(T));
  
  if (isLValType)
    return lval::ConcreteInt(BasicVals.getValue(V));
  else
    return nonlval::ConcreteInt(BasicVals.getValue(V));
}

// Casts.

RVal GRSimpleVals::EvalCast(GRExprEngine& Eng, LVal X, QualType T) {
  
  // Casts from pointers -> pointers, just return the lval.
  //
  // Casts from pointers -> references, just return the lval.  These
  //   can be introduced by the frontend for corner cases, e.g
  //   casting from va_list* to __builtin_va_list&.
  //
  if (LVal::IsLValType(T) || T->isReferenceType())
    return X;
  
  assert (T->isIntegerType());
  
  if (!isa<lval::ConcreteInt>(X))
    return UnknownVal();
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();
  
  llvm::APSInt V = cast<lval::ConcreteInt>(X).getValue();
  V.setIsUnsigned(T->isUnsignedIntegerType() || LVal::IsLValType(T));
  V.extOrTrunc(Eng.getContext().getTypeSize(T));

  return nonlval::ConcreteInt(BasicVals.getValue(V));
}

// Unary operators.

RVal GRSimpleVals::EvalMinus(GRExprEngine& Eng, UnaryOperator* U, NonLVal X){
  
  switch (X.getSubKind()) {
      
    case nonlval::ConcreteIntKind:
      return cast<nonlval::ConcreteInt>(X).EvalMinus(Eng.getBasicVals(), U);
      
    default:
      return UnknownVal();
  }
}

RVal GRSimpleVals::EvalComplement(GRExprEngine& Eng, NonLVal X) {

  switch (X.getSubKind()) {
      
    case nonlval::ConcreteIntKind:
      return cast<nonlval::ConcreteInt>(X).EvalComplement(Eng.getBasicVals());
      
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

RVal GRSimpleVals::DetermEvalBinOpNN(GRStateManager& StateMgr,
                                     BinaryOperator::Opcode Op,
                                     NonLVal L, NonLVal R)  {

  BasicValueFactory& BasicVals = StateMgr.getBasicVals();
  unsigned subkind = L.getSubKind();
  
  while (1) {
    
    switch (subkind) {
      default:
        return UnknownVal();
        
      case nonlval::SymIntConstraintValKind: {
        
        // Logical not?        
        if (!(Op == BinaryOperator::EQ && R.isZeroConstant()))
          return UnknownVal();
        
        const SymIntConstraint& C =
          cast<nonlval::SymIntConstraintVal>(L).getConstraint();
        
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
        
        return nonlval::SymIntConstraintVal(CNew);
      }
        
      case nonlval::ConcreteIntKind:
        
        if (isa<nonlval::ConcreteInt>(R)) {          
          const nonlval::ConcreteInt& L_CI = cast<nonlval::ConcreteInt>(L);
          const nonlval::ConcreteInt& R_CI = cast<nonlval::ConcreteInt>(R);
          return L_CI.EvalBinOp(BasicVals, Op, R_CI);          
        }
        else {
          subkind = R.getSubKind();
          NonLVal tmp = R;
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
        
      case nonlval::SymbolValKind:
        if (isa<nonlval::ConcreteInt>(R)) {
          const SymIntConstraint& C =
            BasicVals.getConstraint(cast<nonlval::SymbolVal>(L).getSymbol(), Op,
                                    cast<nonlval::ConcreteInt>(R).getValue());
          
          return nonlval::SymIntConstraintVal(C);
        }
        else
          return UnknownVal();
    }
  }
}


// Binary Operators (except assignments and comma).

RVal GRSimpleVals::EvalBinOp(GRExprEngine& Eng, BinaryOperator::Opcode Op,
                             LVal L, LVal R) {
  
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

RVal GRSimpleVals::EvalBinOp(GRExprEngine& Eng, BinaryOperator::Opcode Op,
                             LVal L, NonLVal R) {  
  return UnknownVal();
}

// Equality operators for LVals.

RVal GRSimpleVals::EvalEQ(GRExprEngine& Eng, LVal L, LVal R) {
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();
  
  switch (L.getSubKind()) {

    default:
      assert(false && "EQ not implemented for this LVal.");
      return UnknownVal();
      
    case lval::ConcreteIntKind:

      if (isa<lval::ConcreteInt>(R)) {
        bool b = cast<lval::ConcreteInt>(L).getValue() ==
                 cast<lval::ConcreteInt>(R).getValue();
        
        return NonLVal::MakeIntTruthVal(BasicVals, b);
      }
      else if (isa<lval::SymbolVal>(R)) {
        
        const SymIntConstraint& C =
          BasicVals.getConstraint(cast<lval::SymbolVal>(R).getSymbol(),
                               BinaryOperator::EQ,
                               cast<lval::ConcreteInt>(L).getValue());
        
        return nonlval::SymIntConstraintVal(C);
      }
      
      break;
      
    case lval::SymbolValKind: {

      if (isa<lval::ConcreteInt>(R)) {          
        const SymIntConstraint& C =
          BasicVals.getConstraint(cast<lval::SymbolVal>(L).getSymbol(),
                               BinaryOperator::EQ,
                               cast<lval::ConcreteInt>(R).getValue());
        
        return nonlval::SymIntConstraintVal(C);
      }
      
      // FIXME: Implement == for lval Symbols.  This is mainly useful
      //  in iterator loops when traversing a buffer, e.g. while(z != zTerm).
      //  Since this is not useful for many checkers we'll punt on this for 
      //  now.
       
      return UnknownVal();      
    }
      
      // FIXME: Different offsets can map to the same memory cell.
    case lval::ArrayOffsetKind:
    case lval::FieldOffsetKind:
      // Fall-through.
      
    case lval::DeclValKind:
    case lval::FuncValKind:
    case lval::GotoLabelKind:
    case lval::StringLiteralValKind:
      return NonLVal::MakeIntTruthVal(BasicVals, L == R);
  }
  
  return NonLVal::MakeIntTruthVal(BasicVals, false);
}

RVal GRSimpleVals::EvalNE(GRExprEngine& Eng, LVal L, LVal R) {
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();

  switch (L.getSubKind()) {

    default:
      assert(false && "NE not implemented for this LVal.");
      return UnknownVal();
      
    case lval::ConcreteIntKind:
      
      if (isa<lval::ConcreteInt>(R)) {
        bool b = cast<lval::ConcreteInt>(L).getValue() !=
                 cast<lval::ConcreteInt>(R).getValue();
        
        return NonLVal::MakeIntTruthVal(BasicVals, b);
      }
      else if (isa<lval::SymbolVal>(R)) {        
        const SymIntConstraint& C =
          BasicVals.getConstraint(cast<lval::SymbolVal>(R).getSymbol(),
                                  BinaryOperator::NE,
                                  cast<lval::ConcreteInt>(L).getValue());
        
        return nonlval::SymIntConstraintVal(C);
      }
      
      break;
      
    case lval::SymbolValKind: {
      if (isa<lval::ConcreteInt>(R)) {          
        const SymIntConstraint& C =
          BasicVals.getConstraint(cast<lval::SymbolVal>(L).getSymbol(),
                                  BinaryOperator::NE,
                                  cast<lval::ConcreteInt>(R).getValue());
        
        return nonlval::SymIntConstraintVal(C);
      }
      
      // FIXME: Implement != for lval Symbols.  This is mainly useful
      //  in iterator loops when traversing a buffer, e.g. while(z != zTerm).
      //  Since this is not useful for many checkers we'll punt on this for 
      //  now.
      
      return UnknownVal();
      
      break;
    }
      
      // FIXME: Different offsets can map to the same memory cell.
    case lval::ArrayOffsetKind:
    case lval::FieldOffsetKind:
      // Fall-through.
      
    case lval::DeclValKind:
    case lval::FuncValKind:
    case lval::GotoLabelKind:
    case lval::StringLiteralValKind:
      return NonLVal::MakeIntTruthVal(BasicVals, L != R);
  }
  
  return NonLVal::MakeIntTruthVal(BasicVals, true);
}

//===----------------------------------------------------------------------===//
// Transfer function for function calls.
//===----------------------------------------------------------------------===//

void GRSimpleVals::EvalCall(ExplodedNodeSet<GRState>& Dst,
                            GRExprEngine& Eng,
                            GRStmtNodeBuilder<GRState>& Builder,
                            CallExpr* CE, RVal L,
                            ExplodedNode<GRState>* Pred) {
  
  GRStateManager& StateMgr = Eng.getStateManager();
  const GRState* St = Builder.GetState(Pred);
  
  // Invalidate all arguments passed in by reference (LVals).

  for (CallExpr::arg_iterator I = CE->arg_begin(), E = CE->arg_end();
        I != E; ++I) {

    RVal V = StateMgr.GetRVal(St, *I);
    
    if (isa<LVal>(V))
      St = StateMgr.SetRVal(St, cast<LVal>(V), UnknownVal());
    else if (isa<nonlval::LValAsInteger>(V))
      St = StateMgr.SetRVal(St, cast<nonlval::LValAsInteger>(V).getLVal(),
                            UnknownVal());
    
  }
  
  // Make up a symbol for the return value of this function.
  
  if (CE->getType() != Eng.getContext().VoidTy) {    
    unsigned Count = Builder.getCurrentBlockCount();
    SymbolID Sym = Eng.getSymbolManager().getConjuredSymbol(CE, Count);
        
    RVal X = LVal::IsLValType(CE->getType())
             ? cast<RVal>(lval::SymbolVal(Sym)) 
             : cast<RVal>(nonlval::SymbolVal(Sym));
    
    St = StateMgr.SetRVal(St, CE, X, Eng.getCFG().isBlkExpr(CE), false);
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
    
    RVal V = StateMgr.GetRVal(St, *I);
    
    if (isa<LVal>(V))
      St = StateMgr.SetRVal(St, cast<LVal>(V), UnknownVal());
  }
  
  Builder.MakeNode(Dst, ME, Pred, St);
}
