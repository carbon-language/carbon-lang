// BugReporterVisitors.cpp - Helpers for reporting bugs -----------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a set of BugReporter "visitors" which can be used to
//  enhance the diagnostics reported for a bug.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Analysis/PathSensitive/GRState.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

const Stmt *clang::bugreporter::GetDerefExpr(const ExplodedNode *N) {
  // Pattern match for a few useful cases (do something smarter later):
  //   a[0], p->f, *p
  const Stmt *S = N->getLocationAs<PostStmt>()->getStmt();
  
  if (const UnaryOperator *U = dyn_cast<UnaryOperator>(S)) {
    if (U->getOpcode() == UnaryOperator::Deref)
      return U->getSubExpr()->IgnoreParenCasts();
  }
  else if (const MemberExpr *ME = dyn_cast<MemberExpr>(S)) {
    return ME->getBase()->IgnoreParenCasts();
  }
  else if (const ArraySubscriptExpr *AE = dyn_cast<ArraySubscriptExpr>(S)) {
    // Retrieve the base for arrays since BasicStoreManager doesn't know how
    // to reason about them.
    return AE->getBase();
  }
  
  return NULL;  
}

const Stmt*
clang::bugreporter::GetReceiverExpr(const ExplodedNode *N){
  const Stmt *S = N->getLocationAs<PostStmt>()->getStmt();
  if (const ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(S))
    return ME->getReceiver();
  return NULL;
}

const Stmt*
clang::bugreporter::GetDenomExpr(const ExplodedNode *N) {
  const Stmt *S = N->getLocationAs<PostStmt>()->getStmt();
  if (const BinaryOperator *BE = dyn_cast<BinaryOperator>(S))
    return BE->getRHS();
  return NULL;
}

const Stmt*
clang::bugreporter::GetCalleeExpr(const ExplodedNode *N) {
  const Stmt *S = N->getLocationAs<PostStmt>()->getStmt();
  if (const CallExpr *CE = dyn_cast<CallExpr>(S))
    return CE->getCallee();
  return NULL;
}

const Stmt*
clang::bugreporter::GetRetValExpr(const ExplodedNode *N) {
  const Stmt *S = N->getLocationAs<PostStmt>()->getStmt();
  if (const ReturnStmt *RS = dyn_cast<ReturnStmt>(S))
    return RS->getRetValue();
  return NULL;
}

//===----------------------------------------------------------------------===//
// Definitions for bug reporter visitors.
//===----------------------------------------------------------------------===//

namespace {
class VISIBILITY_HIDDEN FindLastStoreBRVisitor : public BugReporterVisitor {
  const MemRegion *R;
  SVal V;
  bool satisfied;
  const ExplodedNode *StoreSite;
public:
  FindLastStoreBRVisitor(SVal v, const MemRegion *r)
  : R(r), V(v), satisfied(false), StoreSite(0) {}
  
  PathDiagnosticPiece* VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext& BRC) {
    
    if (satisfied)
      return NULL;
    
    if (!StoreSite) {      
      const ExplodedNode *Node = N, *Last = NULL;
      
      for ( ; Node ; Last = Node, Node = Node->getFirstPred()) {
        
        if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
          if (const PostStmt *P = Node->getLocationAs<PostStmt>())
            if (const DeclStmt *DS = P->getStmtAs<DeclStmt>())
              if (DS->getSingleDecl() == VR->getDecl()) {
                Last = Node;
                break;
              }
        }
        
        if (Node->getState()->getSVal(R) != V)
          break;
      }
      
      if (!Node || !Last) {
        satisfied = true;
        return NULL;
      }
      
      StoreSite = Last;
    }
    
    if (StoreSite != N)
      return NULL;
    
    satisfied = true;
    std::string sbuf;
    llvm::raw_string_ostream os(sbuf);
    
    if (const PostStmt *PS = N->getLocationAs<PostStmt>()) {
      if (const DeclStmt *DS = PS->getStmtAs<DeclStmt>()) {
        
        if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
          os << "Variable '" << VR->getDecl()->getNameAsString() << "' ";
        }
        else
          return NULL;
        
        if (isa<loc::ConcreteInt>(V)) {
          bool b = false;
          ASTContext &C = BRC.getASTContext();
          if (R->isBoundable()) {
            if (const TypedRegion *TR = dyn_cast<TypedRegion>(R)) {
              if (TR->getValueType(C)->isObjCObjectPointerType()) {
                os << "initialized to nil";
                b = true;
              }
            }
          }
          
          if (!b)
            os << "initialized to a null pointer value";
        }
        else if (isa<nonloc::ConcreteInt>(V)) {
          os << "initialized to " << cast<nonloc::ConcreteInt>(V).getValue();
        }
        else if (V.isUndef()) {
          if (isa<VarRegion>(R)) {
            const VarDecl *VD = cast<VarDecl>(DS->getSingleDecl());
            if (VD->getInit())
              os << "initialized to a garbage value";
            else
              os << "declared without an initial value";              
          }          
        }
      }
    }
    
    if (os.str().empty()) {            
      if (isa<loc::ConcreteInt>(V)) {
        bool b = false;
        ASTContext &C = BRC.getASTContext();
        if (R->isBoundable()) {
          if (const TypedRegion *TR = dyn_cast<TypedRegion>(R)) {
            if (TR->getValueType(C)->isObjCObjectPointerType()) {
              os << "nil object reference stored to ";
              b = true;
            }
          }
        }
        
        if (!b)
          os << "Null pointer value stored to ";
      }
      else if (V.isUndef()) {
        os << "Uninitialized value stored to ";
      }
      else if (isa<nonloc::ConcreteInt>(V)) {
        os << "The value " << cast<nonloc::ConcreteInt>(V).getValue()
           << " is assigned to ";
      }
      else
        return NULL;
      
      if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
        os << '\'' << VR->getDecl()->getNameAsString() << '\'';
      }
      else
        return NULL;
    }
    
    // FIXME: Refactor this into BugReporterContext.
    const Stmt *S = 0;      
    ProgramPoint P = N->getLocation();
    
    if (BlockEdge *BE = dyn_cast<BlockEdge>(&P)) {
      CFGBlock *BSrc = BE->getSrc();
      S = BSrc->getTerminatorCondition();
    }
    else if (PostStmt *PS = dyn_cast<PostStmt>(&P)) {
      S = PS->getStmt();
    }
    
    if (!S)
      return NULL;
    
    // Construct a new PathDiagnosticPiece.
    PathDiagnosticLocation L(S, BRC.getSourceManager());
    return new PathDiagnosticEventPiece(L, os.str());
  }
};


static void registerFindLastStore(BugReporterContext& BRC, const MemRegion *R,
                                  SVal V) {
  BRC.addVisitor(new FindLastStoreBRVisitor(V, R));
}

class VISIBILITY_HIDDEN TrackConstraintBRVisitor : public BugReporterVisitor {
  SVal Constraint;
  const bool Assumption;
  bool isSatisfied;
public:
  TrackConstraintBRVisitor(SVal constraint, bool assumption)
  : Constraint(constraint), Assumption(assumption), isSatisfied(false) {}
  
  PathDiagnosticPiece* VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext& BRC) {
    if (isSatisfied)
      return NULL;
    
    // Check if in the previous state it was feasible for this constraint
    // to *not* be true.
    if (PrevN->getState()->assume(Constraint, !Assumption)) {
      
      isSatisfied = true;
      
      // As a sanity check, make sure that the negation of the constraint
      // was infeasible in the current state.  If it is feasible, we somehow
      // missed the transition point.
      if (N->getState()->assume(Constraint, !Assumption))
        return NULL;
      
      // We found the transition point for the constraint.  We now need to
      // pretty-print the constraint. (work-in-progress)      
      std::string sbuf;
      llvm::raw_string_ostream os(sbuf);
      
      if (isa<Loc>(Constraint)) {
        os << "Assuming pointer value is ";
        os << (Assumption ? "non-null" : "null");
      }
      
      if (os.str().empty())
        return NULL;
      
      // FIXME: Refactor this into BugReporterContext.
      const Stmt *S = 0;      
      ProgramPoint P = N->getLocation();
      
      if (BlockEdge *BE = dyn_cast<BlockEdge>(&P)) {
        CFGBlock *BSrc = BE->getSrc();
        S = BSrc->getTerminatorCondition();
      }
      else if (PostStmt *PS = dyn_cast<PostStmt>(&P)) {
        S = PS->getStmt();
      }
      
      if (!S)
        return NULL;
      
      // Construct a new PathDiagnosticPiece.
      PathDiagnosticLocation L(S, BRC.getSourceManager());
      return new PathDiagnosticEventPiece(L, os.str());
    }
    
    return NULL;
  }  
};
} // end anonymous namespace

static void registerTrackConstraint(BugReporterContext& BRC, SVal Constraint,
                                    bool Assumption) {
  BRC.addVisitor(new TrackConstraintBRVisitor(Constraint, Assumption));  
}

void clang::bugreporter::registerTrackNullOrUndefValue(BugReporterContext& BRC,
                                                       const void *data,
                                                       const ExplodedNode* N) {
  
  const Stmt *S = static_cast<const Stmt*>(data);
  
  if (!S)
    return;
  
  GRStateManager &StateMgr = BRC.getStateManager();
  const GRState *state = N->getState();  
  
  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(S)) {        
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {                
      const VarRegion *R =
      StateMgr.getRegionManager().getVarRegion(VD, N->getLocationContext());
      
      // What did we load?
      SVal V = state->getSVal(S);
      
      if (isa<loc::ConcreteInt>(V) || isa<nonloc::ConcreteInt>(V) 
          || V.isUndef()) {
        registerFindLastStore(BRC, R, V);
      }
    }
  }
  
  SVal V = state->getSValAsScalarOrLoc(S);
  
  // Uncomment this to find cases where we aren't properly getting the
  // base value that was dereferenced.
  // assert(!V.isUnknownOrUndef());
  
  // Is it a symbolic value?
  if (loc::MemRegionVal *L = dyn_cast<loc::MemRegionVal>(&V)) {
    const SubRegion *R = cast<SubRegion>(L->getRegion());
    while (R && !isa<SymbolicRegion>(R)) {
      R = dyn_cast<SubRegion>(R->getSuperRegion());
    }
    
    if (R) {
      assert(isa<SymbolicRegion>(R));
      registerTrackConstraint(BRC, loc::MemRegionVal(R), false);
    }
  }
}
