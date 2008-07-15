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
#include "clang/Analysis/PathSensitive/ValueState.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "llvm/Support/Compiler.h"
#include <sstream>

using namespace clang;

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

template <typename ITERATOR> inline
ExplodedNode<ValueState>* GetNode(ITERATOR I) {
  return *I;
}

template <> inline
ExplodedNode<ValueState>* GetNode(GRExprEngine::undef_arg_iterator I) {
  return I->first;
}

template <typename ITER>
void GenericEmitWarnings(BugReporter& BR, BugType& D, ITER I, ITER E) {
  
  for (; I != E; ++I) {
    BugReport R(D, GetNode(I));    
    BR.EmitWarning(R);
  }
}

//===----------------------------------------------------------------------===//
// Bug Descriptions.
//===----------------------------------------------------------------------===//

namespace {
  
class VISIBILITY_HIDDEN NullDeref : public BugTypeCacheLocation {
public:
  virtual const char* getName() const {
    return "null dereference";
  }
  
  virtual const char* getDescription() const {
    return "Dereference of null pointer.";
  }
  
  virtual void EmitWarnings(BugReporter& BR) {
    GRExprEngine& Eng = cast<GRBugReporter>(BR).getEngine();
    GenericEmitWarnings(BR, *this, Eng.null_derefs_begin(),
                        Eng.null_derefs_end());
  }
};

class VISIBILITY_HIDDEN UndefDeref : public BugTypeCacheLocation {
public:
  virtual const char* getName() const {
    return "bad dereference";
  }
  
  virtual const char* getDescription() const {
    return "Dereference of undefined value.";
  }
  
  virtual void EmitWarnings(BugReporter& BR) {
    GRExprEngine& Eng = cast<GRBugReporter>(BR).getEngine();
    GenericEmitWarnings(BR, *this, Eng.undef_derefs_begin(),
                        Eng.undef_derefs_end());
  }
};
  
class VISIBILITY_HIDDEN UndefBranch : public BugTypeCacheLocation {
public:
  virtual const char* getName() const {
    return "uninitialized value";
  }
  
  virtual const char* getDescription() const {
    return "Branch condition evaluates to an uninitialized value.";
  }
  
  virtual void EmitWarnings(BugReporter& BR);
};
  
class VISIBILITY_HIDDEN DivZero : public BugTypeCacheLocation {
public:
  virtual const char* getName() const {
    return "divide-by-zero";
  }
  
  virtual const char* getDescription() const {
    return "Division by zero/undefined value.";
  }
  
  virtual void EmitWarnings(BugReporter& BR) {
    GRExprEngine& Eng = cast<GRBugReporter>(BR).getEngine();
    GenericEmitWarnings(BR, *this, Eng.explicit_bad_divides_begin(),
                        Eng.explicit_bad_divides_end());
  }
};

class VISIBILITY_HIDDEN UndefResult : public BugTypeCacheLocation {
public:
  virtual const char* getName() const {
    return "undefined result";
  }
  
  virtual const char* getDescription() const {
    return "Result of operation is undefined.";
  }
  
  virtual void EmitWarnings(BugReporter& BR) {
    GRExprEngine& Eng = cast<GRBugReporter>(BR).getEngine();
    GenericEmitWarnings(BR, *this, Eng.undef_results_begin(),
                        Eng.undef_results_end());
  }
};
  
class VISIBILITY_HIDDEN BadCall : public BugTypeCacheLocation {
public:
  virtual const char* getName() const {
    return "invalid function call";
  }
  
  virtual const char* getDescription() const {
    return "Called function is a NULL or undefined function pointer value.";
  }
  
  virtual void EmitWarnings(BugReporter& BR) {
    GRExprEngine& Eng = cast<GRBugReporter>(BR).getEngine();
    GenericEmitWarnings(BR, *this, Eng.bad_calls_begin(),
                        Eng.bad_calls_end());
  }
};
  
  
class VISIBILITY_HIDDEN BadArg : public BugTypeCacheLocation {
public:
  
  virtual ~BadArg() {}
  
  virtual const char* getName() const {
    return "bad argument";
  }
  
  virtual const char* getDescription() const {
    return "Pass-by-value argument in function is undefined.";
  }  

  virtual void EmitWarnings(BugReporter& BR) {
    GRExprEngine& Eng = cast<GRBugReporter>(BR).getEngine();

    for (GRExprEngine::UndefArgsTy::iterator I = Eng.undef_arg_begin(),
         E = Eng.undef_arg_end(); I!=E; ++I) {
      
      // Generate a report for this bug.
      RangedBugReport report(*this, I->first);
      report.addRange(I->second->getSourceRange());

      // Emit the warning.
      BR.EmitWarning(report);
    }

  }
};
  
class VISIBILITY_HIDDEN BadMsgExprArg : public BadArg {
public:
  virtual const char* getName() const {
    return "bad argument";
  }
  
  virtual const char* getDescription() const {
    return "Pass-by-value argument in message expression is undefined.";
  }
  
  virtual void EmitWarnings(BugReporter& BR) {
    GRExprEngine& Eng = cast<GRBugReporter>(BR).getEngine();
    
    for (GRExprEngine::UndefArgsTy::iterator I=Eng.msg_expr_undef_arg_begin(),
         E = Eng.msg_expr_undef_arg_end(); I!=E; ++I) {
      
      // Generate a report for this bug.
      RangedBugReport report(*this, I->first);
      report.addRange(I->second->getSourceRange());
      
      // Emit the warning.
      BR.EmitWarning(report);
    }    
  }
};

class VISIBILITY_HIDDEN BadReceiver : public BugTypeCacheLocation {
public:  
  virtual const char* getName() const {
    return "bad receiver";
  }
  
  virtual const char* getDescription() const {
    return "Receiver in message expression is an uninitialized value.";
  }
  
  virtual void EmitWarnings(BugReporter& BR) {
    GRExprEngine& Eng = cast<GRBugReporter>(BR).getEngine();
    
    for (GRExprEngine::UndefReceiversTy::iterator I=Eng.undef_receivers_begin(),
         End = Eng.undef_receivers_end(); I!=End; ++I) {
          
      // Generate a report for this bug.
      RangedBugReport report(*this, *I);
      
      ExplodedNode<ValueState>* N = *I;
      Stmt *S = cast<PostStmt>(N->getLocation()).getStmt();
      Expr* E = cast<ObjCMessageExpr>(S)->getReceiver();
      assert (E && "Receiver cannot be NULL");
      report.addRange(E->getSourceRange());
      
      // Emit the warning.
      BR.EmitWarning(report);
    }    
  }
};
  
class VISIBILITY_HIDDEN RetStack : public BugTypeCacheLocation {
public:
  virtual const char* getName() const {
    return "return of stack address";
  }
  
  virtual const char* getDescription() const {
    return "Address of stack-allocated variable returned.";
  }
  
  virtual void EmitWarnings(BugReporter& BR) {
    GRExprEngine& Eng = cast<GRBugReporter>(BR).getEngine();
    GenericEmitWarnings(BR, *this, Eng.ret_stackaddr_begin(),
                        Eng.ret_stackaddr_end());
  }
};
  
} // end anonymous namespace


namespace {

struct VISIBILITY_HIDDEN FindUndefExpr {
  ValueStateManager& VM;
  const ValueState* St;
  
  FindUndefExpr(ValueStateManager& V, const ValueState* S) : VM(V), St(S) {}
  
  Expr* FindExpr(Expr* Ex) {
    
    if (!MatchesCriteria(Ex))
      return 0;    
        
    for (Stmt::child_iterator I=Ex->child_begin(), E=Ex->child_end(); I!=E; ++I)
      if (Expr* ExI = dyn_cast_or_null<Expr>(*I)) {
        Expr* E2 = FindExpr(ExI);
        if (E2) return E2;
      }
    
    return Ex;
  }
  
  bool MatchesCriteria(Expr* Ex) { return VM.GetRVal(St, Ex).isUndef(); }  
};
  
} // end anonymous namespace
  
  
void UndefBranch::EmitWarnings(BugReporter& BR) {

  GRExprEngine& Eng = cast<GRBugReporter>(BR).getEngine();
  
  for (GRExprEngine::undef_branch_iterator I=Eng.undef_branches_begin(),
       E=Eng.undef_branches_end(); I!=E; ++I) {

    // What's going on here: we want to highlight the subexpression of the
    // condition that is the most likely source of the "uninitialized
    // branch condition."  We do a recursive walk of the condition's
    // subexpressions and roughly look for the most nested subexpression
    // that binds to Undefined.  We then highlight that expression's range.
    
    BlockEdge B = cast<BlockEdge>((*I)->getLocation());
    Expr* Ex = cast<Expr>(B.getSrc()->getTerminatorCondition());
    assert (Ex && "Block must have a terminator.");
    
    // Get the predecessor node and check if is a PostStmt with the Stmt
    // being the terminator condition.  We want to inspect the state
    // of that node instead because it will contain main information about
    // the subexpressions.
    
    assert (!(*I)->pred_empty());
    
    // Note: any predecessor will do.  They should have identical state,
    // since all the BlockEdge did was act as an error sink since the value
    // had to already be undefined.
    ExplodedNode<ValueState> *N = *(*I)->pred_begin();
    ProgramPoint P = N->getLocation();

    const ValueState* St = (*I)->getState();
    
    if (PostStmt* PS = dyn_cast<PostStmt>(&P))
      if (PS->getStmt() == Ex)
        St = N->getState();
        
    FindUndefExpr FindIt(Eng.getStateManager(), St);
    Ex = FindIt.FindExpr(Ex);
    
    RangedBugReport R(*this, *I);
    R.addRange(Ex->getSourceRange());
    
    BR.EmitWarning(R);
  }
}


void GRSimpleVals::RegisterChecks(GRExprEngine& Eng) {
  
  // Path-sensitive checks.
  Eng.Register(new NullDeref());
  Eng.Register(new UndefDeref());
  Eng.Register(new UndefBranch());
  Eng.Register(new DivZero());
  Eng.Register(new UndefResult());
  Eng.Register(new BadCall());
  Eng.Register(new RetStack());
  Eng.Register(new BadArg());
  Eng.Register(new BadMsgExprArg());
  Eng.Register(new BadReceiver());
  
  // Add extra checkers.
  ASTContext& Ctx = Eng.getContext();
  ValueStateManager* VMgr = &Eng.getStateManager();

  GRSimpleAPICheck* Check = CreateBasicObjCFoundationChecks(Ctx, VMgr);
  Eng.AddCheck(Check, Stmt::ObjCMessageExprClass);
  
  Check = CreateAuditCFNumberCreate(Ctx, VMgr);
  Eng.AddCheck(Check, Stmt::CallExprClass);
  
}

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

  // Only handle casts from integers to integers.
  if (!T->isIntegerType())
    return UnknownVal();
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();
  
  llvm::APSInt V = cast<nonlval::ConcreteInt>(X).getValue();
  V.setIsUnsigned(T->isUnsignedIntegerType() || LVal::IsLValType(T));
  V.extOrTrunc(Eng.getContext().getTypeSize(T));
  
  if (LVal::IsLValType(T))
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

RVal GRSimpleVals::EvalBinOp(GRExprEngine& Eng, BinaryOperator::Opcode Op,
                             NonLVal L, NonLVal R)  {
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();
  
  while (1) {
    
    switch (L.getSubKind()) {
      default:
        return UnknownVal();
        
      case nonlval::ConcreteIntKind:
        
        if (isa<nonlval::ConcreteInt>(R)) {          
          const nonlval::ConcreteInt& L_CI = cast<nonlval::ConcreteInt>(L);
          const nonlval::ConcreteInt& R_CI = cast<nonlval::ConcreteInt>(R);          
          return L_CI.EvalBinOp(BasicVals, Op, R_CI);          
        }
        else {
          NonLVal tmp = R;
          R = L;
          L = tmp;
          continue;
        }
        
      case nonlval::SymbolValKind: {
        
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

void GRSimpleVals::EvalCall(ExplodedNodeSet<ValueState>& Dst,
                            GRExprEngine& Eng,
                            GRStmtNodeBuilder<ValueState>& Builder,
                            CallExpr* CE, RVal L,
                            ExplodedNode<ValueState>* Pred) {
  
  ValueStateManager& StateMgr = Eng.getStateManager();
  const ValueState* St = Builder.GetState(Pred);
  
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

void GRSimpleVals::EvalObjCMessageExpr(ExplodedNodeSet<ValueState>& Dst,
                                       GRExprEngine& Eng,
                                       GRStmtNodeBuilder<ValueState>& Builder,
                                       ObjCMessageExpr* ME,
                                       ExplodedNode<ValueState>* Pred) {
  
  
  // The basic transfer function logic for message expressions does nothing.
  // We just invalidate all arguments passed in by references.
  
  ValueStateManager& StateMgr = Eng.getStateManager();
  const ValueState* St = Builder.GetState(Pred);
  
  for (ObjCMessageExpr::arg_iterator I = ME->arg_begin(), E = ME->arg_end();
       I != E; ++I) {
    
    RVal V = StateMgr.GetRVal(St, *I);
    
    if (isa<LVal>(V))
      St = StateMgr.SetRVal(St, cast<LVal>(V), UnknownVal());
  }
  
  Builder.MakeNode(Dst, ME, Pred, St);
}
