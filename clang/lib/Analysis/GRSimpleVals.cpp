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
#include "clang/Analysis/PathSensitive/ValueState.h"
#include "clang/Analysis/PathDiagnostic.h"
#include <sstream>

using namespace clang;

//===----------------------------------------------------------------------===//
// Bug Descriptions.
//===----------------------------------------------------------------------===//

namespace bugdesc {

struct NullDeref {
  static const char* getName() { return "Null dereference"; }

  static PathDiagnosticPiece* getEndPath(SourceManager& SMgr,
                                         ExplodedNode<ValueState> *N);
};
  
PathDiagnosticPiece* NullDeref::getEndPath(SourceManager& SMgr,
                                           ExplodedNode<ValueState> *N) {
  
  Expr* E = cast<Expr>(cast<PostStmt>(N->getLocation()).getStmt());
  
  // FIXME: Do better ranges for different kinds of null dereferences.

  FullSourceLoc L(E->getLocStart(), SMgr);
  
  PathDiagnosticPiece* P = new PathDiagnosticPiece(L, getName());
  P->addRange(E->getSourceRange());
  
  return P;
}
  
} // end namespace: bugdesc
  
//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//
  
template <typename ITERATOR>
static inline ExplodedNode<ValueState>* GetNode(ITERATOR I) {
  return *I;
}

template <>
static inline ExplodedNode<ValueState>*
GetNode(GRExprEngine::undef_arg_iterator I) {
  return I->first;
}
  
template <typename ITERATOR>
static inline ProgramPoint GetLocation(ITERATOR I) {
  return (*I)->getLocation();
}
  
template <>
static inline ProgramPoint GetLocation(GRExprEngine::undef_arg_iterator I) {
  return I->first->getLocation();
}
  
static inline Stmt* GetStmt(const ProgramPoint& P) {
  if (const PostStmt* PS = dyn_cast<PostStmt>(&P)) {
    return PS->getStmt();
  }
  else if (const BlockEdge* BE = dyn_cast<BlockEdge>(&P)) {
    return BE->getSrc()->getTerminator();
  }
  else if (const BlockEntrance* BE = dyn_cast<BlockEntrance>(&P)) {
    return BE->getFirstStmt();
  }

  assert (false && "Unsupported ProgramPoint.");
  return NULL;
}

//===----------------------------------------------------------------------===//
// Pathless Warnings
//===----------------------------------------------------------------------===//
  
template <typename ITERATOR>
static void EmitDiag(Diagnostic& Diag, PathDiagnosticClient* PD,
                     SourceManager& SrcMgr,
unsigned ErrorDiag, ITERATOR I) {  
  
  Stmt* S = GetStmt(GetLocation(I));
  SourceRange R = S->getSourceRange();
  Diag.Report(PD, FullSourceLoc(S->getLocStart(), SrcMgr), ErrorDiag,
              NULL, 0, &R, 1);    
}


template <>
static void EmitDiag(Diagnostic& Diag, PathDiagnosticClient* PD, 
                     SourceManager& SrcMgr, unsigned ErrorDiag,
                     GRExprEngine::undef_arg_iterator I) {

  Stmt* S1 = GetStmt(GetLocation(I));
  Expr* E2 = cast<Expr>(I->second);
  
  SourceLocation Loc = S1->getLocStart();
  SourceRange R = E2->getSourceRange();
  Diag.Report(PD, FullSourceLoc(Loc, SrcMgr), ErrorDiag, 0, 0, &R, 1);
}

template <typename ITERATOR>
static void EmitWarning(Diagnostic& Diag,  PathDiagnosticClient* PD,
                        SourceManager& SrcMgr,
                        ITERATOR I, ITERATOR E, const char* msg) {
 
  std::ostringstream Out;
  std::string Str(msg);

  if (!PD) {
    Out << "[CHECKER] " << msg;
    Str = Out.str();
    msg = Str.c_str();
  }
  
  bool isFirst = true;
  unsigned ErrorDiag = 0;
  llvm::SmallPtrSet<void*,10> CachedErrors;  
  
  for (; I != E; ++I) {
  
    if (isFirst) {
      isFirst = false;    
      ErrorDiag = Diag.getCustomDiagID(Diagnostic::Warning, msg);
    }
    else {
      
      // HACK: Cache the location of the error.  Don't emit the same
      // warning for the same error type that occurs at the same program
      // location but along a different path.
      void* p = GetLocation(I).getRawData();

      if (CachedErrors.count(p))
        continue;
      
      CachedErrors.insert(p);
    }
    
    EmitDiag(Diag, PD, SrcMgr, ErrorDiag, I);  
  }
}
  
//===----------------------------------------------------------------------===//
// Path warnings.
//===----------------------------------------------------------------------===//

static void GeneratePathDiagnostic(PathDiagnostic& PD,
                                   ASTContext& Ctx,
                                   ExplodedNode<ValueState>* N) {
  
  SourceManager& SMgr = Ctx.getSourceManager();
  
  while (!N->pred_empty()) {
  
    ExplodedNode<ValueState>* LastNode = N;
    N = *(N->pred_begin());
    
    ProgramPoint P = N->getLocation();
    
    if (const BlockEdge* BE = dyn_cast<BlockEdge>(&P)) {
      
      CFGBlock* Src = BE->getSrc();
      CFGBlock* Dst = BE->getDst();
      
      Stmt* T = Src->getTerminator();
      
      if (!T)
        continue;

      FullSourceLoc L(T->getLocStart(), SMgr);
      
      switch (T->getStmtClass()) {
        default:
          break;
        
        case Stmt::GotoStmtClass:
        case Stmt::IndirectGotoStmtClass: {
          
          Stmt* S = GetStmt(LastNode->getLocation());
 
          if (!S)
            continue;
          
          std::ostringstream os;

          os << "Control jumps to line "
             << SMgr.getLogicalLineNumber(S->getLocStart()) << ".\n";
          
          PD.push_front(new PathDiagnosticPiece(L, os.str()));
          break;
        }
          
        case Stmt::SwitchStmtClass: {
          
          // Figure out what case arm we took.
          
          Stmt* S = Dst->getLabel();

          if (!S)
            continue;
          
          std::ostringstream os;

          switch (S->getStmtClass()) {
            default:
              continue;
              
            case Stmt::DefaultStmtClass: {
                          
              os << "Control jumps to the 'default' case at line "
                  << SMgr.getLogicalLineNumber(S->getLocStart()) << ".\n";
              
              break;
            }
              
            case Stmt::CaseStmtClass: {
              
              os << "Control jumps to 'case ";
                            
              Expr* CondE = cast<SwitchStmt>(T)->getCond();
              unsigned bits = Ctx.getTypeSize(CondE->getType());
              
              llvm::APSInt V1(bits, false);
              
              CaseStmt* Case = cast<CaseStmt>(S);

              if (!Case->getLHS()->isIntegerConstantExpr(V1, Ctx, 0, true)) {
                assert (false &&
                        "Case condition must evaluate to an integer constant.");
                continue;
              }
              
              os << V1.toString();
                
              // Get the RHS of the case, if it exists.
                
              if (Expr* E = Case->getRHS()) {
              
                llvm::APSInt V2(bits, false);

                if (!E->isIntegerConstantExpr(V2, Ctx, 0, true)) {
                  assert (false &&
                "Case condition (RHS) must evaluate to an integer constant.");
                  continue;
                }
                  
                os << " .. " << V2.toString();
              }
              
              os << ":'  at line " 
                 << SMgr.getLogicalLineNumber(S->getLocStart()) << ".\n";

              break;
              
            }
          }
          
          PD.push_front(new PathDiagnosticPiece(L, os.str()));
          break;
        }
        
        
        case Stmt::DoStmtClass:
        case Stmt::WhileStmtClass:
        case Stmt::ForStmtClass:
        case Stmt::IfStmtClass: {
        
          if (*(Src->succ_begin()+1) == Dst)
            PD.push_front(new PathDiagnosticPiece(L, "Taking false branch."));
          else 
            PD.push_front(new PathDiagnosticPiece(L, "Taking true branch."));
          
          break;
        }
      }
    }  
  }
}

template <typename ITERATOR, typename DESC>
static void Report(PathDiagnosticClient& PDC, ASTContext& Ctx, DESC,
                   ITERATOR I, ITERATOR E) {
  

  if (I == E)
    return;
  
  const char* BugName = DESC::getName();

  llvm::SmallPtrSet<void*,10> CachedErrors;  
  
  for (; I != E; ++I) {
    
    // HACK: Cache the location of the error.  Don't emit the same
    // warning for the same error type that occurs at the same program
    // location but along a different path.
    void* p = GetLocation(I).getRawData();
    
    if (CachedErrors.count(p))
      continue;
    
    CachedErrors.insert(p);
    
    // Create the PathDiagnostic.
    
    PathDiagnostic D(BugName);
    
    // Get the end-of-path diagnostic.
    D.push_back(DESC::getEndPath(Ctx.getSourceManager(), GetNode(I)));
    
    // Generate the rest of the diagnostic.
    GeneratePathDiagnostic(D, Ctx, GetNode(I));
    
    PDC.HandlePathDiagnostic(D);
  }
}


//===----------------------------------------------------------------------===//
// Analysis Driver.
//===----------------------------------------------------------------------===//
  
namespace clang {
  
unsigned RunGRSimpleVals(CFG& cfg, Decl& CD, ASTContext& Ctx,
                                Diagnostic& Diag, PathDiagnosticClient* PD,
                                bool Visualize, bool TrimGraph) {
  
  GRCoreEngine<GRExprEngine> Eng(cfg, CD, Ctx);
  GRExprEngine* CS = &Eng.getCheckerState();
  
  // Set base transfer functions.
  GRSimpleVals GRSV;
  CS->setTransferFunctions(GRSV);
  
  // Add extra checkers.
  llvm::OwningPtr<GRSimpleAPICheck> FoundationCheck(
    CreateBasicObjCFoundationChecks(Ctx, &CS->getStateManager()));
  
  CS->AddObjCMessageExprCheck(FoundationCheck.get());
  
  // Execute the worklist algorithm.
  Eng.ExecuteWorkList(120000);
  
  SourceManager& SrcMgr = Ctx.getSourceManager();  

  if (!PD)
    EmitWarning(Diag, PD, SrcMgr,
                CS->null_derefs_begin(), CS->null_derefs_end(),
                "Dereference of NULL pointer.");
  else 
    Report(*PD, Ctx, bugdesc::NullDeref(),
           CS->null_derefs_begin(), CS->null_derefs_end());

  
  EmitWarning(Diag, PD, SrcMgr,
              CS->undef_derefs_begin(),
              CS->undef_derefs_end(),
              "Dereference of undefined value.");
  
  EmitWarning(Diag, PD, SrcMgr,
              CS->undef_branches_begin(),
              CS->undef_branches_end(),
              "Branch condition evaluates to an uninitialized value.");
  
  EmitWarning(Diag, PD, SrcMgr,
              CS->explicit_bad_divides_begin(),
              CS->explicit_bad_divides_end(),
              "Division by zero/undefined value.");
  
  EmitWarning(Diag, PD, SrcMgr,
              CS->undef_results_begin(),
              CS->undef_results_end(),
              "Result of operation is undefined.");
  
  EmitWarning(Diag, PD, SrcMgr,
              CS->bad_calls_begin(),
              CS->bad_calls_end(),
              "Call using a NULL or undefined function pointer value.");
  
  EmitWarning(Diag, PD, SrcMgr,
              CS->undef_arg_begin(),
              CS->undef_arg_end(),
              "Pass-by-value argument in function is undefined.");
  
  EmitWarning(Diag, PD, SrcMgr,
              CS->msg_expr_undef_arg_begin(),
              CS->msg_expr_undef_arg_end(),
              "Pass-by-value argument in message expression is undefined.");
  
  EmitWarning(Diag, PD, SrcMgr,
              CS->undef_receivers_begin(),
              CS->undef_receivers_end(),
              "Receiver in message expression is an uninitialized value.");
  
  EmitWarning(Diag, PD, SrcMgr,
              CS->ret_stackaddr_begin(),
              CS->ret_stackaddr_end(),
              "Address of stack-allocated variable returned.");

  FoundationCheck.get()->ReportResults(Diag);
#ifndef NDEBUG
  if (Visualize) CS->ViewGraph(TrimGraph);
#endif
  
  return Eng.getGraph().size();
}
  
} // end clang namespace

//===----------------------------------------------------------------------===//
// Transfer function for Casts.
//===----------------------------------------------------------------------===//

RVal GRSimpleVals::EvalCast(GRExprEngine& Eng, NonLVal X, QualType T) {
  
  if (!isa<nonlval::ConcreteInt>(X))
    return UnknownVal();

  BasicValueFactory& BasicVals = Eng.getBasicVals();
  
  llvm::APSInt V = cast<nonlval::ConcreteInt>(X).getValue();
  V.setIsUnsigned(T->isUnsignedIntegerType() || T->isPointerType() 
                  || T->isObjCQualifiedIdType());
  V.extOrTrunc(Eng.getContext().getTypeSize(T));
  
  if (T->isPointerType())
    return lval::ConcreteInt(BasicVals.getValue(V));
  else
    return nonlval::ConcreteInt(BasicVals.getValue(V));
}

// Casts.

RVal GRSimpleVals::EvalCast(GRExprEngine& Eng, LVal X, QualType T) {
  
  if (T->isPointerLikeType() || T->isObjCQualifiedIdType())
    return X;
  
  assert (T->isIntegerType());
  
  if (!isa<lval::ConcreteInt>(X))
    return UnknownVal();
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();
  
  llvm::APSInt V = cast<lval::ConcreteInt>(X).getValue();
  V.setIsUnsigned(T->isUnsignedIntegerType() || T->isPointerType());
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
      
    case lval::DeclValKind:
    case lval::FuncValKind:
    case lval::GotoLabelKind:
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
      
    case lval::DeclValKind:
    case lval::FuncValKind:
    case lval::GotoLabelKind:
      return NonLVal::MakeIntTruthVal(BasicVals, L != R);
  }
  
  return NonLVal::MakeIntTruthVal(BasicVals, true);
}

//===----------------------------------------------------------------------===//
// Transfer function for Function Calls.
//===----------------------------------------------------------------------===//

void GRSimpleVals::EvalCall(ExplodedNodeSet<ValueState>& Dst,
                            GRExprEngine& Eng,
                            GRStmtNodeBuilder<ValueState>& Builder,
                            CallExpr* CE, LVal L,
                            ExplodedNode<ValueState>* Pred) {
  
  ValueStateManager& StateMgr = Eng.getStateManager();
  ValueState* St = Builder.GetState(Pred);
  
  // Invalidate all arguments passed in by reference (LVals).

  for (CallExpr::arg_iterator I = CE->arg_begin(), E = CE->arg_end();
        I != E; ++I) {

    RVal V = StateMgr.GetRVal(St, *I);
    
    if (isa<LVal>(V))
      St = StateMgr.SetRVal(St, cast<LVal>(V), UnknownVal());
  }
  
  // Make up a symbol for the return value of this function.
  
  if (CE->getType() != Eng.getContext().VoidTy) {    
    unsigned Count = Builder.getCurrentBlockCount();
    SymbolID Sym = Eng.getSymbolManager().getConjuredSymbol(CE, Count);
        
    RVal X = CE->getType()->isPointerType() 
             ? cast<RVal>(lval::SymbolVal(Sym)) 
             : cast<RVal>(nonlval::SymbolVal(Sym));
    
    St = StateMgr.SetRVal(St, CE, X, Eng.getCFG().isBlkExpr(CE), false);
  }  
    
  Builder.MakeNode(Dst, CE, Pred, St);
}
