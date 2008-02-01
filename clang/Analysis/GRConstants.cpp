//===-- GRConstants.cpp - Simple, Path-Sens. Constant Prop. ------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//               Constant Propagation via Graph Reachability
//
//  This files defines a simple analysis that performs path-sensitive
//  constant propagation within a function.  An example use of this analysis
//  is to perform simple checks for NULL dereferences.
//
//===----------------------------------------------------------------------===//

#include "RValues.h"
#include "ValueState.h"

#include "clang/Analysis/PathSensitive/GREngine.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ASTContext.h"
#include "clang/Analysis/Analyses/LiveVariables.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"

#include <functional>

#ifndef NDEBUG
#include "llvm/Support/GraphWriter.h"
#include <sstream>
#endif

using namespace clang;
using llvm::dyn_cast;
using llvm::cast;
using llvm::APSInt;

//===----------------------------------------------------------------------===//
// The Checker.
//
//  FIXME: This checker logic should be eventually broken into two components.
//         The first is the "meta"-level checking logic; the code that
//         does the Stmt visitation, fetching values from the map, etc.
//         The second part does the actual state manipulation.  This way we
//         get more of a separate of concerns of these two pieces, with the
//         latter potentially being refactored back into the main checking
//         logic.
//===----------------------------------------------------------------------===//

namespace {
  
class VISIBILITY_HIDDEN GRConstants {
    
public:
  typedef ValueState StateTy;
  typedef GRStmtNodeBuilder<GRConstants> StmtNodeBuilder;
  typedef GRBranchNodeBuilder<GRConstants> BranchNodeBuilder;
  typedef ExplodedGraph<GRConstants> GraphTy;
  typedef GraphTy::NodeTy NodeTy;
  
  class NodeSet {
    typedef llvm::SmallVector<NodeTy*,3> ImplTy;
    ImplTy Impl;
  public:
    
    NodeSet() {}
    NodeSet(NodeTy* N) { assert (N && !N->isSink()); Impl.push_back(N); }
    
    void Add(NodeTy* N) { if (N && !N->isSink()) Impl.push_back(N); }
    
    typedef ImplTy::iterator       iterator;
    typedef ImplTy::const_iterator const_iterator;
        
    unsigned size() const { return Impl.size(); }
    bool empty() const { return Impl.empty(); }
    
    iterator begin() { return Impl.begin(); }
    iterator end()   { return Impl.end(); }

    const_iterator begin() const { return Impl.begin(); }
    const_iterator end() const { return Impl.end(); }
  };
                                                              
protected:
  /// G - the simulation graph.
  GraphTy& G;
  
  /// Liveness - live-variables information the ValueDecl* and block-level
  ///  Expr* in the CFG.  Used to prune out dead state.
  LiveVariables Liveness;

  /// Builder - The current GRStmtNodeBuilder which is used when building the nodes
  ///  for a given statement.
  StmtNodeBuilder* Builder;
  
  /// StateMgr - Object that manages the data for all created states.
  StateTy::Factory StateMgr;
  
  /// ValueMgr - Object that manages the data for all created RValues.
  ValueManager ValMgr;
  
  /// SymMgr - Object that manages the symbol information.
  SymbolManager SymMgr;
  
  /// StmtEntryNode - The immediate predecessor node.
  NodeTy* StmtEntryNode;
  
  /// CurrentStmt - The current block-level statement.
  Stmt* CurrentStmt;
  
  /// UninitBranches - Nodes in the ExplodedGraph that result from
  ///  taking a branch based on an uninitialized value.
  typedef llvm::SmallPtrSet<NodeTy*,5> UninitBranchesTy;
  UninitBranchesTy UninitBranches;
  
  bool StateCleaned;
  
  ASTContext& getContext() const { return G.getContext(); }
  
public:
  GRConstants(GraphTy& g) : G(g), Liveness(G.getCFG(), G.getFunctionDecl()),
      Builder(NULL), ValMgr(G.getContext()), StmtEntryNode(NULL),
      CurrentStmt(NULL) {
    
    // Compute liveness information.
    Liveness.runOnCFG(G.getCFG());
    Liveness.runOnAllBlocks(G.getCFG(), NULL, true);
  }
  
  /// getCFG - Returns the CFG associated with this analysis.
  CFG& getCFG() { return G.getCFG(); }
  
  /// getInitialState - Return the initial state used for the root vertex
  ///  in the ExplodedGraph.
  StateTy getInitialState() {
    StateTy St = StateMgr.GetEmptyMap();
    
    // Iterate the parameters.
    FunctionDecl& F = G.getFunctionDecl();
    
    for (FunctionDecl::param_iterator I=F.param_begin(), E=F.param_end(); 
          I!=E; ++I)
      St = SetValue(St, LValueDecl(*I), RValue::GetSymbolValue(SymMgr, *I));
    
    return St;
  }
  
  bool isUninitControlFlow(const NodeTy* N) const {
    return N->isSink() && UninitBranches.count(const_cast<NodeTy*>(N)) != 0;
  }

  /// ProcessStmt - Called by GREngine. Used to generate new successor
  ///  nodes by processing the 'effects' of a block-level statement.
  void ProcessStmt(Stmt* S, StmtNodeBuilder& builder);    
  
  /// ProcessBranch - Called by GREngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a branch condition.
  void ProcessBranch(Stmt* Condition, Stmt* Term, BranchNodeBuilder& builder);

  /// RemoveDeadBindings - Return a new state that is the same as 'M' except
  ///  that all subexpression mappings are removed and that any
  ///  block-level expressions that are not live at 'S' also have their
  ///  mappings removed.
  StateTy RemoveDeadBindings(Stmt* S, StateTy M);

  StateTy SetValue(StateTy St, Stmt* S, const RValue& V);  

  StateTy SetValue(StateTy St, const Stmt* S, const RValue& V) {
    return SetValue(St, const_cast<Stmt*>(S), V);
  }
  
  StateTy SetValue(StateTy St, const LValue& LV, const RValue& V);
  
  RValue GetValue(const StateTy& St, Stmt* S);  
  inline RValue GetValue(const StateTy& St, const Stmt* S) {
    return GetValue(St, const_cast<Stmt*>(S));
  }
  
  RValue GetValue(const StateTy& St, const LValue& LV);
  LValue GetLValue(const StateTy& St, Stmt* S);
    
  /// Assume - Create new state by assuming that a given expression
  ///  is true or false.
  inline StateTy Assume(StateTy St, RValue Cond, bool Assumption, 
                        bool& isFeasible) {
    if (isa<LValue>(Cond))
      return Assume(St, cast<LValue>(Cond), Assumption, isFeasible);
    else
      return Assume(St, cast<NonLValue>(Cond), Assumption, isFeasible);
  }
  
  StateTy Assume(StateTy St, LValue Cond, bool Assumption, bool& isFeasible);
  StateTy Assume(StateTy St, NonLValue Cond, bool Assumption, bool& isFeasible);
  
  void Nodify(NodeSet& Dst, Stmt* S, NodeTy* Pred, StateTy St);
  
  /// Visit - Transfer function logic for all statements.  Dispatches to
  ///  other functions that handle specific kinds of statements.
  void Visit(Stmt* S, NodeTy* Pred, NodeSet& Dst);

  /// VisitCast - Transfer function logic for all casts (implicit and explicit).
  void VisitCast(Expr* CastE, Expr* E, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitUnaryOperator - Transfer function logic for unary operators.
  void VisitUnaryOperator(UnaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitBinaryOperator - Transfer function logic for binary operators.
  void VisitBinaryOperator(BinaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitDeclStmt - Transfer function logic for DeclStmts.
  void VisitDeclStmt(DeclStmt* DS, NodeTy* Pred, NodeSet& Dst);
};
} // end anonymous namespace


void GRConstants::ProcessBranch(Stmt* Condition, Stmt* Term,
                                BranchNodeBuilder& builder) {

  StateTy PrevState = builder.getState();
  
  // Remove old bindings for subexpressions.  
  for (StateTy::iterator I=PrevState.begin(), E=PrevState.end(); I!=E; ++I)
    if (I.getKey().isSubExpr())
      PrevState = StateMgr.Remove(PrevState, I.getKey());
  
  RValue V = GetValue(PrevState, Condition);
  
  switch (V.getBaseKind()) {
    default:
      break;

    case RValue::InvalidKind:
      builder.generateNode(PrevState, true);
      builder.generateNode(PrevState, false);
      return;
      
    case RValue::UninitializedKind: {      
      NodeTy* N = builder.generateNode(PrevState, true);

      if (N) {
        N->markAsSink();
        UninitBranches.insert(N);
      }
      
      builder.markInfeasible(false);
      return;
    }      
  }

  // Process the true branch.
  bool isFeasible = true;
  StateTy St = Assume(PrevState, V, true, isFeasible);

  if (isFeasible) builder.generateNode(St, true);
  else {
    builder.markInfeasible(true);
    isFeasible = true;
  }
  
  // Process the false branch.  
  St = Assume(PrevState, V, false, isFeasible);
  
  if (isFeasible) builder.generateNode(St, false);
  else builder.markInfeasible(false);

}

void GRConstants::ProcessStmt(Stmt* S, StmtNodeBuilder& builder) {
  Builder = &builder;

  StmtEntryNode = builder.getLastNode();
  CurrentStmt = S;
  NodeSet Dst;
  StateCleaned = false;

  Visit(S, StmtEntryNode, Dst);

  // If no nodes were generated, generate a new node that has all the
  // dead mappings removed.
  if (Dst.size() == 1 && *Dst.begin() == StmtEntryNode) {
    StateTy St = RemoveDeadBindings(S, StmtEntryNode->getState());
    builder.generateNode(S, St, StmtEntryNode);
  }
  
  CurrentStmt = NULL;
  StmtEntryNode = NULL;
  Builder = NULL;
}


RValue GRConstants::GetValue(const StateTy& St, const LValue& LV) {
  switch (LV.getSubKind()) {
    case LValueDeclKind: {
      StateTy::TreeTy* T = St.SlimFind(cast<LValueDecl>(LV).getDecl()); 
      return T ? T->getValue().second : InvalidValue();
    }
    default:
      assert (false && "Invalid LValue.");
      break;
  }
  
  return InvalidValue();
}
  
RValue GRConstants::GetValue(const StateTy& St, Stmt* S) {
  for (;;) {
    switch (S->getStmtClass()) {
        
      // ParenExprs are no-ops.
        
      case Stmt::ParenExprClass:
        S = cast<ParenExpr>(S)->getSubExpr();
        continue;
        
      // DeclRefExprs can either evaluate to an LValue or a Non-LValue
      // (assuming an implicit "load") depending on the context.  In this
      // context we assume that we are retrieving the value contained
      // within the referenced variables.
        
      case Stmt::DeclRefExprClass:
        return GetValue(St, LValueDecl(cast<DeclRefExpr>(S)->getDecl()));

      // Integer literals evaluate to an RValue.  Simply retrieve the
      // RValue for the literal.
        
      case Stmt::IntegerLiteralClass:
        return NonLValue::GetValue(ValMgr, cast<IntegerLiteral>(S));
              
      // Casts where the source and target type are the same
      // are no-ops.  We blast through these to get the descendant
      // subexpression that has a value.
        
      case Stmt::ImplicitCastExprClass: {
        ImplicitCastExpr* C = cast<ImplicitCastExpr>(S);
        if (C->getType() == C->getSubExpr()->getType()) {
          S = C->getSubExpr();
          continue;
        }
        break;
      }

      case Stmt::CastExprClass: {
        CastExpr* C = cast<CastExpr>(S);
        if (C->getType() == C->getSubExpr()->getType()) {
          S = C->getSubExpr();
          continue;
        }
        break;
      }

      // Handle all other Stmt* using a lookup.

      default:
        break;
    };
    
    break;
  }
  
  StateTy::TreeTy* T = St.SlimFind(S);
    
  return T ? T->getValue().second : InvalidValue();
}

LValue GRConstants::GetLValue(const StateTy& St, Stmt* S) {
  while (ParenExpr* P = dyn_cast<ParenExpr>(S))
    S = P->getSubExpr();
  
  if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(S))
    return LValueDecl(DR->getDecl());
    
  return cast<LValue>(GetValue(St, S));
}


GRConstants::StateTy GRConstants::SetValue(StateTy St, Stmt* S,
                                           const RValue& V) {
  assert (S);
  
  if (!StateCleaned) {
    St = RemoveDeadBindings(CurrentStmt, St);
    StateCleaned = true;
  }
  
  bool isBlkExpr = false;
  
  if (S == CurrentStmt) {
    isBlkExpr = getCFG().isBlkExpr(S);

    if (!isBlkExpr)
      return St;
  }
  
  return V.isValid() ? StateMgr.Add(St, ValueKey(S,isBlkExpr), V)
                     : St;
}

GRConstants::StateTy GRConstants::SetValue(StateTy St, const LValue& LV,
                                           const RValue& V) {
  if (!LV.isValid())
    return St;
  
  if (!StateCleaned) {
    St = RemoveDeadBindings(CurrentStmt, St);
    StateCleaned = true;
  }

  switch (LV.getSubKind()) {
    case LValueDeclKind:        
      return V.isValid() ? StateMgr.Add(St, cast<LValueDecl>(LV).getDecl(), V)
                         : StateMgr.Remove(St, cast<LValueDecl>(LV).getDecl());
      
    default:
      assert ("SetValue for given LValue type not yet implemented.");
      return St;
  }
}

GRConstants::StateTy GRConstants::RemoveDeadBindings(Stmt* Loc, StateTy M) {
  // Note: in the code below, we can assign a new map to M since the
  //  iterators are iterating over the tree of the *original* map.
  StateTy::iterator I = M.begin(), E = M.end();


  for (; I!=E && !I.getKey().isSymbol(); ++I) {
    // Remove old bindings for subexpressions and "dead" 
    // block-level expressions.    
    if (I.getKey().isSubExpr() ||
        I.getKey().isBlkExpr() && !Liveness.isLive(Loc,cast<Stmt>(I.getKey()))){
      M = StateMgr.Remove(M, I.getKey());
    }
    else if (I.getKey().isDecl()) { // Remove bindings for "dead" decls.
      if (VarDecl* V = dyn_cast<VarDecl>(cast<ValueDecl>(I.getKey())))
        if (!Liveness.isLive(Loc, V))
          M = StateMgr.Remove(M, I.getKey());
    }
  }

  return M;
}

void GRConstants::Nodify(NodeSet& Dst, Stmt* S, GRConstants::NodeTy* Pred, 
                         GRConstants::StateTy St) {
 
  // If the state hasn't changed, don't generate a new node.
  if (St == Pred->getState())
    return;
  
  Dst.Add(Builder->generateNode(S, St, Pred));
}

void GRConstants::VisitCast(Expr* CastE, Expr* E, GRConstants::NodeTy* Pred,
                            GRConstants::NodeSet& Dst) {
  
  QualType T = CastE->getType();

  // Check for redundant casts.
  if (E->getType() == T) {
    Dst.Add(Pred);
    return;
  }
  
  NodeSet S1;
  Visit(E, Pred, S1);
  
  for (NodeSet::iterator I1=S1.begin(), E1=S1.end(); I1 != E1; ++I1) {
    NodeTy* N = *I1;
    StateTy St = N->getState();
    const RValue& V = GetValue(St, E);
    Nodify(Dst, CastE, N, SetValue(St, CastE, V.Cast(ValMgr, CastE)));
  }
}

void GRConstants::VisitDeclStmt(DeclStmt* DS, GRConstants::NodeTy* Pred,
                                GRConstants::NodeSet& Dst) {
  
  StateTy St = Pred->getState();
  
  for (const ScopedDecl* D = DS->getDecl(); D; D = D->getNextDeclarator())
    if (const VarDecl* VD = dyn_cast<VarDecl>(D)) {
      const Expr* E = VD->getInit();      
      St = SetValue(St, LValueDecl(VD),
                    E ? GetValue(St, E) : UninitializedValue());
    }

  Nodify(Dst, DS, Pred, St);
  
  if (Dst.empty())
    Dst.Add(Pred);  
}

void GRConstants::VisitUnaryOperator(UnaryOperator* U,
                                     GRConstants::NodeTy* Pred,
                                     GRConstants::NodeSet& Dst) {
  NodeSet S1;
  Visit(U->getSubExpr(), Pred, S1);
    
  for (NodeSet::iterator I1=S1.begin(), E1=S1.end(); I1 != E1; ++I1) {
    NodeTy* N1 = *I1;
    StateTy St = N1->getState();
    
    switch (U->getOpcode()) {
      case UnaryOperator::PostInc: {
        const LValue& L1 = GetLValue(St, U->getSubExpr());
        NonLValue R1 = cast<NonLValue>(GetValue(St, L1));
        NonLValue R2 = NonLValue::GetValue(ValMgr, 1U, U->getType(),
                                           U->getLocStart());
        
        NonLValue Result = R1.Add(ValMgr, R2);
        Nodify(Dst, U, N1, SetValue(SetValue(St, U, R1), L1, Result));
        break;
      }
        
      case UnaryOperator::PostDec: {
        const LValue& L1 = GetLValue(St, U->getSubExpr());
        NonLValue R1 = cast<NonLValue>(GetValue(St, L1));
        NonLValue R2 = NonLValue::GetValue(ValMgr, 1U, U->getType(),
                                           U->getLocStart());
        
        NonLValue Result = R1.Sub(ValMgr, R2);
        Nodify(Dst, U, N1, SetValue(SetValue(St, U, R1), L1, Result));
        break;
      }
        
      case UnaryOperator::PreInc: {
        const LValue& L1 = GetLValue(St, U->getSubExpr());
        NonLValue R1 = cast<NonLValue>(GetValue(St, L1));
        NonLValue R2 = NonLValue::GetValue(ValMgr, 1U, U->getType(),
                                           U->getLocStart());        
        
        NonLValue Result = R1.Add(ValMgr, R2);
        Nodify(Dst, U, N1, SetValue(SetValue(St, U, Result), L1, Result));
        break;
      }
        
      case UnaryOperator::PreDec: {
        const LValue& L1 = GetLValue(St, U->getSubExpr());
        NonLValue R1 = cast<NonLValue>(GetValue(St, L1));
        NonLValue R2 = NonLValue::GetValue(ValMgr, 1U, U->getType(),
                                           U->getLocStart());
        
        NonLValue Result = R1.Sub(ValMgr, R2);
        Nodify(Dst, U, N1, SetValue(SetValue(St, U, Result), L1, Result));
        break;
      }
        
      case UnaryOperator::Minus: {
        const NonLValue& R1 = cast<NonLValue>(GetValue(St, U->getSubExpr()));
        Nodify(Dst, U, N1, SetValue(St, U, R1.UnaryMinus(ValMgr, U)));
        break;
      }
        
      case UnaryOperator::AddrOf: {
        const LValue& L1 = GetLValue(St, U->getSubExpr());
        Nodify(Dst, U, N1, SetValue(St, U, L1));
        break;
      }
        
      case UnaryOperator::Deref: {
        const LValue& L1 = GetLValue(St, U->getSubExpr());
        Nodify(Dst, U, N1, SetValue(St, U, GetValue(St, L1)));
        break;
      }
        
      default: ;
        assert (false && "Not implemented.");
    }    
  }
}

void GRConstants::VisitBinaryOperator(BinaryOperator* B,
                                      GRConstants::NodeTy* Pred,
                                      GRConstants::NodeSet& Dst) {
  NodeSet S1;
  Visit(B->getLHS(), Pred, S1);

  for (NodeSet::iterator I1=S1.begin(), E1=S1.end(); I1 != E1; ++I1) {
    NodeTy* N1 = *I1;
    
    // When getting the value for the LHS, check if we are in an assignment.
    // In such cases, we want to (initially) treat the LHS as an LValue,
    // so we use GetLValue instead of GetValue so that DeclRefExpr's are
    // evaluated to LValueDecl's instead of to an NonLValue.
    const RValue& V1 = 
      B->isAssignmentOp() ? GetLValue(N1->getState(), B->getLHS())
                          : GetValue(N1->getState(), B->getLHS());
    
    NodeSet S2;
    Visit(B->getRHS(), N1, S2);
  
    for (NodeSet::iterator I2=S2.begin(), E2=S2.end(); I2 != E2; ++I2) {
      NodeTy* N2 = *I2;
      StateTy St = N2->getState();
      const RValue& V2 = GetValue(St, B->getRHS());

      switch (B->getOpcode()) {
        default: 
          Dst.Add(N2);
          break;
          
        // Arithmetic opreators.
          
        case BinaryOperator::Add: {
          const NonLValue& R1 = cast<NonLValue>(V1);
          const NonLValue& R2 = cast<NonLValue>(V2);
          
          Nodify(Dst, B, N2, SetValue(St, B, R1.Add(ValMgr, R2)));
          break;
        }

        case BinaryOperator::Sub: {
          const NonLValue& R1 = cast<NonLValue>(V1);
          const NonLValue& R2 = cast<NonLValue>(V2);
	        Nodify(Dst, B, N2, SetValue(St, B, R1.Sub(ValMgr, R2)));
          break;
        }
          
        case BinaryOperator::Mul: {
          const NonLValue& R1 = cast<NonLValue>(V1);
          const NonLValue& R2 = cast<NonLValue>(V2);
	        Nodify(Dst, B, N2, SetValue(St, B, R1.Mul(ValMgr, R2)));
          break;
        }
          
        case BinaryOperator::Div: {
          const NonLValue& R1 = cast<NonLValue>(V1);
          const NonLValue& R2 = cast<NonLValue>(V2);
	        Nodify(Dst, B, N2, SetValue(St, B, R1.Div(ValMgr, R2)));
          break;
        }
          
        case BinaryOperator::Rem: {
          const NonLValue& R1 = cast<NonLValue>(V1);
          const NonLValue& R2 = cast<NonLValue>(V2);
	        Nodify(Dst, B, N2, SetValue(St, B, R1.Rem(ValMgr, R2)));
          break;
        }
          
        // Assignment operators.
          
        case BinaryOperator::Assign: {
          const LValue& L1 = cast<LValue>(V1);
          const NonLValue& R2 = cast<NonLValue>(V2);
          Nodify(Dst, B, N2, SetValue(SetValue(St, B, R2), L1, R2));
          break;
        }
          
        case BinaryOperator::AddAssign: {
          const LValue& L1 = cast<LValue>(V1);
          NonLValue R1 = cast<NonLValue>(GetValue(N1->getState(), L1));
          NonLValue Result = R1.Add(ValMgr, cast<NonLValue>(V2));
          Nodify(Dst, B, N2, SetValue(SetValue(St, B, Result), L1, Result));
          break;
        }
          
        case BinaryOperator::SubAssign: {
          const LValue& L1 = cast<LValue>(V1);
          NonLValue R1 = cast<NonLValue>(GetValue(N1->getState(), L1));
          NonLValue Result = R1.Sub(ValMgr, cast<NonLValue>(V2));
          Nodify(Dst, B, N2, SetValue(SetValue(St, B, Result), L1, Result));
          break;
        }
          
        case BinaryOperator::MulAssign: {
          const LValue& L1 = cast<LValue>(V1);
          NonLValue R1 = cast<NonLValue>(GetValue(N1->getState(), L1));
          NonLValue Result = R1.Mul(ValMgr, cast<NonLValue>(V2));
          Nodify(Dst, B, N2, SetValue(SetValue(St, B, Result), L1, Result));
          break;
        }
          
        case BinaryOperator::DivAssign: {
          const LValue& L1 = cast<LValue>(V1);
          NonLValue R1 = cast<NonLValue>(GetValue(N1->getState(), L1));
          NonLValue Result = R1.Div(ValMgr, cast<NonLValue>(V2));
          Nodify(Dst, B, N2, SetValue(SetValue(St, B, Result), L1, Result));
          break;
        }
          
        case BinaryOperator::RemAssign: {
          const LValue& L1 = cast<LValue>(V1);
          NonLValue R1 = cast<NonLValue>(GetValue(N1->getState(), L1));
          NonLValue Result = R1.Rem(ValMgr, cast<NonLValue>(V2));
          Nodify(Dst, B, N2, SetValue(SetValue(St, B, Result), L1, Result));
          break;
        }
          
        // Equality operators.

        case BinaryOperator::EQ:
          // FIXME: should we allow XX.EQ() to return a set of values,
          //  allowing state bifurcation?  In such cases, they will also
          //  modify the state (meaning that a new state will be returned
          //  as well).
          assert (B->getType() == getContext().IntTy);
          
          if (isa<LValue>(V1)) {
            const LValue& L1 = cast<LValue>(V1);
            const LValue& L2 = cast<LValue>(V2);
            St = SetValue(St, B, L1.EQ(ValMgr, L2));
          }
          else {
            const NonLValue& R1 = cast<NonLValue>(V1);
            const NonLValue& R2 = cast<NonLValue>(V2);
            St = SetValue(St, B, R1.EQ(ValMgr, R2));
          }
          
          Nodify(Dst, B, N2, St);
          break;
      }
    }
  }
}


void GRConstants::Visit(Stmt* S, GRConstants::NodeTy* Pred,
                        GRConstants::NodeSet& Dst) {

  // FIXME: add metadata to the CFG so that we can disable
  //  this check when we KNOW that there is no block-level subexpression.
  //  The motivation is that this check requires a hashtable lookup.

  if (S != CurrentStmt && getCFG().isBlkExpr(S)) {
    Dst.Add(Pred);
    return;
  }

  switch (S->getStmtClass()) {
    case Stmt::BinaryOperatorClass:
    case Stmt::CompoundAssignOperatorClass:
      VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Dst);
      break;
      
    case Stmt::UnaryOperatorClass:
      VisitUnaryOperator(cast<UnaryOperator>(S), Pred, Dst);
      break;
      
    case Stmt::ParenExprClass:
      Visit(cast<ParenExpr>(S)->getSubExpr(), Pred, Dst);
      break;
      
    case Stmt::ImplicitCastExprClass: {
      ImplicitCastExpr* C = cast<ImplicitCastExpr>(S);
      VisitCast(C, C->getSubExpr(), Pred, Dst);
      break;
    }
      
    case Stmt::CastExprClass: {
      CastExpr* C = cast<CastExpr>(S);
      VisitCast(C, C->getSubExpr(), Pred, Dst);
      break;
    }
      
    case Stmt::DeclStmtClass:
      VisitDeclStmt(cast<DeclStmt>(S), Pred, Dst);
      break;
      
    default:
      Dst.Add(Pred); // No-op. Simply propagate the current state unchanged.
      break;
  }
}

//===----------------------------------------------------------------------===//
// "Assume" logic.
//===----------------------------------------------------------------------===//

GRConstants::StateTy GRConstants::Assume(StateTy St, LValue Cond, bool Assumption, 
                                         bool& isFeasible) {    
  
  switch (Cond.getSubKind()) {
    default:
      assert (false && "'Assume' not implemented for this NonLValue.");
      return St;
      
    case LValueDeclKind:
      isFeasible = Assumption;
      return St;
      
    case ConcreteIntLValueKind: {
      bool b = cast<ConcreteIntLValue>(Cond).getValue() != 0;
      isFeasible = b ? Assumption : !Assumption;      
      return St;
    }
  }
}

GRConstants::StateTy GRConstants::Assume(StateTy St, NonLValue Cond, bool Assumption, 
                                         bool& isFeasible) {
  
  switch (Cond.getSubKind()) {
    default:
      assert (false && "'Assume' not implemented for this NonLValue.");
      return St;
      
    case ConcreteIntKind: {
      bool b = cast<ConcreteInt>(Cond).getValue() != 0;
      isFeasible = b ? Assumption : !Assumption;      
      return St;
    }
  }
}


//===----------------------------------------------------------------------===//
// Driver.
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
static GRConstants* GraphPrintCheckerState;

namespace llvm {
template<>
struct VISIBILITY_HIDDEN DOTGraphTraits<GRConstants::NodeTy*> :
  public DefaultDOTGraphTraits {

  static void PrintKindLabel(std::ostream& Out, ValueKey::Kind kind) {
    switch (kind) {
      case ValueKey::IsSubExpr:  Out << "Sub-Expressions:\\l"; break;
      case ValueKey::IsDecl:    Out << "Variables:\\l"; break;
      case ValueKey::IsBlkExpr: Out << "Block-level Expressions:\\l"; break;
      default: assert (false && "Unknown ValueKey type.");
    }
  }
    
  static void PrintKind(std::ostream& Out, GRConstants::StateTy M,
                        ValueKey::Kind kind, bool isFirstGroup = false) {
    bool isFirst = true;
    
    for (GRConstants::StateTy::iterator I=M.begin(), E=M.end();I!=E;++I) {        
      if (I.getKey().getKind() != kind)
        continue;
    
      if (isFirst) {
        if (!isFirstGroup) Out << "\\l\\l";
        PrintKindLabel(Out, kind);
        isFirst = false;
      }
      else
        Out << "\\l";
      
      Out << ' ';
    
      if (ValueDecl* V = dyn_cast<ValueDecl>(I.getKey()))
        Out << V->getName();          
      else {
        Stmt* E = cast<Stmt>(I.getKey());
        Out << " (" << (void*) E << ") ";
        E->printPretty(Out);
      }
    
      Out << " : ";
      I.getData().print(Out);
    }
  }
    
  static std::string getNodeLabel(const GRConstants::NodeTy* N, void*) {
    std::ostringstream Out;

    // Program Location.
    ProgramPoint Loc = N->getLocation();
    
    switch (Loc.getKind()) {
      case ProgramPoint::BlockEntranceKind:
        Out << "Block Entrance: B" 
            << cast<BlockEntrance>(Loc).getBlock()->getBlockID();
        break;
      
      case ProgramPoint::BlockExitKind:
        assert (false);
        break;
        
      case ProgramPoint::PostStmtKind: {
        const PostStmt& L = cast<PostStmt>(Loc);
        Out << L.getStmt()->getStmtClassName() << ':' 
            << (void*) L.getStmt() << ' ';
        
        L.getStmt()->printPretty(Out);
        break;
      }
    
      default: {
        const BlockEdge& E = cast<BlockEdge>(Loc);
        Out << "Edge: (B" << E.getSrc()->getBlockID() << ", B"
            << E.getDst()->getBlockID()  << ')';
        
        if (Stmt* T = E.getSrc()->getTerminator()) {
          Out << "\\|Terminator: ";
          E.getSrc()->printTerminator(Out);
          
          if (isa<SwitchStmt>(T)) {
            // FIXME
          }
          else {
            Out << "\\lCondition: ";
            if (*E.getSrc()->succ_begin() == E.getDst())
              Out << "true";
            else
              Out << "false";                        
          }
          
          Out << "\\l";
        }
        
        if (GraphPrintCheckerState->isUninitControlFlow(N)) {
          Out << "\\|Control-flow based on\\lUninitialized value.\\l";
        }
      }
    }
    
    Out << "\\|StateID: " << (void*) N->getState().getRoot() << "\\|";
    
    PrintKind(Out, N->getState(), ValueKey::IsDecl, true);
    PrintKind(Out, N->getState(), ValueKey::IsBlkExpr);
    PrintKind(Out, N->getState(), ValueKey::IsSubExpr);
      
    Out << "\\l";
    return Out.str();
  }
};
} // end llvm namespace    
#endif

namespace clang {
void RunGRConstants(CFG& cfg, FunctionDecl& FD, ASTContext& Ctx) {
  GREngine<GRConstants> Engine(cfg, FD, Ctx);
  Engine.ExecuteWorkList();  
#ifndef NDEBUG
  GraphPrintCheckerState = &Engine.getCheckerState();
  llvm::ViewGraph(*Engine.getGraph().roots_begin(),"GRConstants");
  GraphPrintCheckerState = NULL;
#endif  
}
} // end clang namespace
