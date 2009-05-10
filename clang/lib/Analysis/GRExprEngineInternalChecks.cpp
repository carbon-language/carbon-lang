//=-- GRExprEngineInternalChecks.cpp - Builtin GRExprEngine Checks---*- C++ -*-=
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the BugType classes used by GRExprEngine to report
//  bugs derived from builtin checks in the path-sensitive engine.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

template <typename ITERATOR> inline
ExplodedNode<GRState>* GetNode(ITERATOR I) {
  return *I;
}

template <> inline
ExplodedNode<GRState>* GetNode(GRExprEngine::undef_arg_iterator I) {
  return I->first;
}

//===----------------------------------------------------------------------===//
// Forward declarations for bug reporter visitors.
//===----------------------------------------------------------------------===//

static void registerTrackNullValue(BugReporterContext& BRC,
                                   const ExplodedNode<GRState>* N);

//===----------------------------------------------------------------------===//
// Bug Descriptions.
//===----------------------------------------------------------------------===//

namespace {

class VISIBILITY_HIDDEN BuiltinBugReport : public BugReport {
public:
  BuiltinBugReport(BugType& bt, const char* desc,
                   const ExplodedNode<GRState> *n)
  : BugReport(bt, desc, n) {}
  
  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode<GRState>* N);
};  
  
class VISIBILITY_HIDDEN BuiltinBug : public BugType {
  GRExprEngine &Eng;
protected:
  const std::string desc;
public:
  BuiltinBug(GRExprEngine *eng, const char* n, const char* d)
    : BugType(n, "Logic Errors"), Eng(*eng), desc(d) {}

  BuiltinBug(GRExprEngine *eng, const char* n)
    : BugType(n, "Logic Errors"), Eng(*eng), desc(n) {}
  
  virtual void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) = 0;

  void FlushReports(BugReporter& BR) { FlushReportsImpl(BR, Eng); }
  
  virtual void registerInitialVisitors(BugReporterContext& BRC,
                                       const ExplodedNode<GRState>* N,
                                       BuiltinBugReport *R) {}
  
  template <typename ITER> void Emit(BugReporter& BR, ITER I, ITER E);
};
  
  
template <typename ITER>
void BuiltinBug::Emit(BugReporter& BR, ITER I, ITER E) {
  for (; I != E; ++I) BR.EmitReport(new BuiltinBugReport(*this, desc.c_str(),
                                                         GetNode(I)));
}  

void BuiltinBugReport::registerInitialVisitors(BugReporterContext& BRC,
                                               const ExplodedNode<GRState>* N) {
  static_cast<BuiltinBug&>(getBugType()).registerInitialVisitors(BRC, N, this);
}  
  
class VISIBILITY_HIDDEN NullDeref : public BuiltinBug {
public:
  NullDeref(GRExprEngine* eng)
    : BuiltinBug(eng,"Null dereference", "Dereference of null pointer") {}

  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    Emit(BR, Eng.null_derefs_begin(), Eng.null_derefs_end());
  }
  
  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode<GRState>* N,
                               BuiltinBugReport *R) {
    registerTrackNullValue(BRC, N);
  }
};
  
class VISIBILITY_HIDDEN NilReceiverStructRet : public BugType {
  GRExprEngine &Eng;
public:
  NilReceiverStructRet(GRExprEngine* eng) :
    BugType("'nil' receiver with struct return type", "Logic Errors"),  
    Eng(*eng) {}

  void FlushReports(BugReporter& BR) {
    for (GRExprEngine::nil_receiver_struct_ret_iterator
          I=Eng.nil_receiver_struct_ret_begin(),
          E=Eng.nil_receiver_struct_ret_end(); I!=E; ++I) {

      std::string sbuf;
      llvm::raw_string_ostream os(sbuf);
      PostStmt P = cast<PostStmt>((*I)->getLocation());
      ObjCMessageExpr *ME = cast<ObjCMessageExpr>(P.getStmt());
      os << "The receiver in the message expression is 'nil' and results in the"
            " returned value (of type '"
         << ME->getType().getAsString()
         << "') to be garbage or otherwise undefined.";

      RangedBugReport *R = new RangedBugReport(*this, os.str().c_str(), *I);
      R->addRange(ME->getReceiver()->getSourceRange());
      BR.EmitReport(R);
    }
  }
};

class VISIBILITY_HIDDEN NilReceiverLargerThanVoidPtrRet : public BugType {
  GRExprEngine &Eng;
public:
  NilReceiverLargerThanVoidPtrRet(GRExprEngine* eng) :
  BugType("'nil' receiver with return type larger than sizeof(void *)", 
          "Logic Errors"),  
  Eng(*eng) {}
  
  void FlushReports(BugReporter& BR) {
    for (GRExprEngine::nil_receiver_larger_than_voidptr_ret_iterator
         I=Eng.nil_receiver_larger_than_voidptr_ret_begin(),
         E=Eng.nil_receiver_larger_than_voidptr_ret_end(); I!=E; ++I) {
      
      std::string sbuf;
      llvm::raw_string_ostream os(sbuf);
      PostStmt P = cast<PostStmt>((*I)->getLocation());
      ObjCMessageExpr *ME = cast<ObjCMessageExpr>(P.getStmt());
      os << "The receiver in the message expression is 'nil' and results in the"
      " returned value (of type '"
      << ME->getType().getAsString()
      << "' and of size "
      << Eng.getContext().getTypeSize(ME->getType()) / 8
      << " bytes) to be garbage or otherwise undefined.";
      
      RangedBugReport *R = new RangedBugReport(*this, os.str().c_str(), *I);
      R->addRange(ME->getReceiver()->getSourceRange());
      BR.EmitReport(R);
    }
  }
};
  
class VISIBILITY_HIDDEN UndefinedDeref : public BuiltinBug {
public:
  UndefinedDeref(GRExprEngine* eng)
    : BuiltinBug(eng,"Dereference of undefined pointer value") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    Emit(BR, Eng.undef_derefs_begin(), Eng.undef_derefs_end());
  }
};

class VISIBILITY_HIDDEN DivZero : public BuiltinBug {
public:
  DivZero(GRExprEngine* eng)
    : BuiltinBug(eng,"Division-by-zero",
                 "Division by zero or undefined value.") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    Emit(BR, Eng.explicit_bad_divides_begin(), Eng.explicit_bad_divides_end());
  }
};
  
class VISIBILITY_HIDDEN UndefResult : public BuiltinBug {
public:
  UndefResult(GRExprEngine* eng) : BuiltinBug(eng,"Undefined result",
                             "Result of operation is undefined.") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    Emit(BR, Eng.undef_results_begin(), Eng.undef_results_end());
  }
};
  
class VISIBILITY_HIDDEN BadCall : public BuiltinBug {
public:
  BadCall(GRExprEngine *eng)
  : BuiltinBug(eng, "Invalid function call",
        "Called function pointer is a null or undefined pointer value") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    Emit(BR, Eng.bad_calls_begin(), Eng.bad_calls_end());
  }
};

class VISIBILITY_HIDDEN BadArg : public BuiltinBug {
public:
  BadArg(GRExprEngine* eng) : BuiltinBug(eng,"Uninitialized argument",  
    "Pass-by-value argument in function call is undefined.") {}

  BadArg(GRExprEngine* eng, const char* d)
    : BuiltinBug(eng,"Uninitialized argument", d) {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    for (GRExprEngine::UndefArgsTy::iterator I = Eng.undef_arg_begin(),
         E = Eng.undef_arg_end(); I!=E; ++I) {
      // Generate a report for this bug.
      RangedBugReport *report = new RangedBugReport(*this, desc.c_str(),
                                                    I->first);
      report->addRange(I->second->getSourceRange());
      BR.EmitReport(report);
    }
  }
};
  
class VISIBILITY_HIDDEN BadMsgExprArg : public BadArg {
public:
  BadMsgExprArg(GRExprEngine* eng) 
    : BadArg(eng,"Pass-by-value argument in message expression is undefined"){}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    for (GRExprEngine::UndefArgsTy::iterator I=Eng.msg_expr_undef_arg_begin(),
         E = Eng.msg_expr_undef_arg_end(); I!=E; ++I) {      
      // Generate a report for this bug.
      RangedBugReport *report = new RangedBugReport(*this, desc.c_str(),
                                                    I->first);
      report->addRange(I->second->getSourceRange());
      BR.EmitReport(report);
    }    
  }
};
  
class VISIBILITY_HIDDEN BadReceiver : public BuiltinBug {
public:  
  BadReceiver(GRExprEngine* eng)
  : BuiltinBug(eng,"Uninitialized receiver",
               "Receiver in message expression is an uninitialized value") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    for (GRExprEngine::ErrorNodes::iterator I=Eng.undef_receivers_begin(),
         End = Eng.undef_receivers_end(); I!=End; ++I) {
      
      // Generate a report for this bug.
      RangedBugReport *report = new RangedBugReport(*this, desc.c_str(), *I);
      ExplodedNode<GRState>* N = *I;
      Stmt *S = cast<PostStmt>(N->getLocation()).getStmt();
      Expr* E = cast<ObjCMessageExpr>(S)->getReceiver();
      assert (E && "Receiver cannot be NULL");
      report->addRange(E->getSourceRange());
      BR.EmitReport(report);
    }    
  }
};

class VISIBILITY_HIDDEN RetStack : public BuiltinBug {
public:
  RetStack(GRExprEngine* eng)
    : BuiltinBug(eng, "Return of address to stack-allocated memory") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    for (GRExprEngine::ret_stackaddr_iterator I=Eng.ret_stackaddr_begin(),
         End = Eng.ret_stackaddr_end(); I!=End; ++I) {

      ExplodedNode<GRState>* N = *I;
      Stmt *S = cast<PostStmt>(N->getLocation()).getStmt();
      Expr* E = cast<ReturnStmt>(S)->getRetValue();
      assert (E && "Return expression cannot be NULL");
      
      // Get the value associated with E.
      loc::MemRegionVal V =
        cast<loc::MemRegionVal>(Eng.getStateManager().GetSVal(N->getState(),
                                                               E));
      
      // Generate a report for this bug.
      std::string buf;
      llvm::raw_string_ostream os(buf);
      SourceRange R;
      
      // Check if the region is a compound literal.
      if (const CompoundLiteralRegion* CR = 
            dyn_cast<CompoundLiteralRegion>(V.getRegion())) {
        
        const CompoundLiteralExpr* CL = CR->getLiteralExpr();
        os << "Address of stack memory associated with a compound literal "
              "declared on line "
            << BR.getSourceManager()
                    .getInstantiationLineNumber(CL->getLocStart())
            << " returned.";
        
        R = CL->getSourceRange();
      }
      else if (const AllocaRegion* AR = dyn_cast<AllocaRegion>(V.getRegion())) {
        const Expr* ARE = AR->getExpr();
        SourceLocation L = ARE->getLocStart();
        R = ARE->getSourceRange();
        
        os << "Address of stack memory allocated by call to alloca() on line "
           << BR.getSourceManager().getInstantiationLineNumber(L)
           << " returned.";
      }      
      else {        
        os << "Address of stack memory associated with local variable '"
           << V.getRegion()->getString() << "' returned.";
      }
      
      RangedBugReport *report = new RangedBugReport(*this, os.str().c_str(), N);
      report->addRange(E->getSourceRange());
      if (R.isValid()) report->addRange(R);
      BR.EmitReport(report);
    }
  }
};
  
class VISIBILITY_HIDDEN RetUndef : public BuiltinBug {
public:
  RetUndef(GRExprEngine* eng) : BuiltinBug(eng, "Uninitialized return value",
              "Uninitialized or undefined return value returned to caller.") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    Emit(BR, Eng.ret_undef_begin(), Eng.ret_undef_end());
  }
};

class VISIBILITY_HIDDEN UndefBranch : public BuiltinBug {
  struct VISIBILITY_HIDDEN FindUndefExpr {
    GRStateManager& VM;
    const GRState* St;
    
    FindUndefExpr(GRStateManager& V, const GRState* S) : VM(V), St(S) {}
    
    Expr* FindExpr(Expr* Ex) {      
      if (!MatchesCriteria(Ex))
        return 0;
      
      for (Stmt::child_iterator I=Ex->child_begin(), E=Ex->child_end();I!=E;++I)
        if (Expr* ExI = dyn_cast_or_null<Expr>(*I)) {
          Expr* E2 = FindExpr(ExI);
          if (E2) return E2;
        }
      
      return Ex;
    }
    
    bool MatchesCriteria(Expr* Ex) { return VM.GetSVal(St, Ex).isUndef(); }
  };
  
public:
  UndefBranch(GRExprEngine *eng)
    : BuiltinBug(eng,"Use of uninitialized value",
                 "Branch condition evaluates to an uninitialized value.") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
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
      ExplodedNode<GRState> *N = *(*I)->pred_begin();
      ProgramPoint P = N->getLocation();
      const GRState* St = (*I)->getState();

      if (PostStmt* PS = dyn_cast<PostStmt>(&P))
        if (PS->getStmt() == Ex)
          St = N->getState();

      FindUndefExpr FindIt(Eng.getStateManager(), St);
      Ex = FindIt.FindExpr(Ex);

      RangedBugReport *R = new RangedBugReport(*this, desc.c_str(), *I);
      R->addRange(Ex->getSourceRange());
      BR.EmitReport(R);
    }
  }
};

class VISIBILITY_HIDDEN OutOfBoundMemoryAccess : public BuiltinBug {
public:
  OutOfBoundMemoryAccess(GRExprEngine* eng)
    : BuiltinBug(eng,"Out-of-bounds memory access",
                     "Load or store into an out-of-bound memory position.") {}

  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    Emit(BR, Eng.explicit_oob_memacc_begin(), Eng.explicit_oob_memacc_end());
  }
};
  
class VISIBILITY_HIDDEN BadSizeVLA : public BuiltinBug {
public:
  BadSizeVLA(GRExprEngine* eng) :
    BuiltinBug(eng, "Bad variable-length array (VLA) size") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    for (GRExprEngine::ErrorNodes::iterator
          I = Eng.ExplicitBadSizedVLA.begin(),
          E = Eng.ExplicitBadSizedVLA.end(); I!=E; ++I) {

      // Determine whether this was a 'zero-sized' VLA or a VLA with an
      // undefined size.
      GRExprEngine::NodeTy* N = *I;
      PostStmt PS = cast<PostStmt>(N->getLocation());      
      DeclStmt *DS = cast<DeclStmt>(PS.getStmt());
      VarDecl* VD = cast<VarDecl>(*DS->decl_begin());
      QualType T = Eng.getContext().getCanonicalType(VD->getType());
      VariableArrayType* VT = cast<VariableArrayType>(T);
      Expr* SizeExpr = VT->getSizeExpr();
      
      std::string buf;
      llvm::raw_string_ostream os(buf);
      os << "The expression used to specify the number of elements in the "
            "variable-length array (VLA) '"
         << VD->getNameAsString() << "' evaluates to ";
      
      bool isUndefined = Eng.getStateManager().GetSVal(N->getState(),
                                                       SizeExpr).isUndef();
      
      if (isUndefined)
        os << "an undefined or garbage value.";
      else
        os << "0. VLAs with no elements have undefined behavior.";
      
      std::string shortBuf;
      llvm::raw_string_ostream os_short(shortBuf);
      os_short << "Variable-length array '" << VD->getNameAsString() << "' "
               << (isUndefined ? "garbage value for array size"
                   : "has zero elements (undefined behavior)");

      RangedBugReport *report = new RangedBugReport(*this,
                                                    os_short.str().c_str(),
                                                    os.str().c_str(), N);
      report->addRange(SizeExpr->getSourceRange());
      BR.EmitReport(report);
    }
  }
};

//===----------------------------------------------------------------------===//
// __attribute__(nonnull) checking

class VISIBILITY_HIDDEN CheckAttrNonNull : public GRSimpleAPICheck {
  BugType *BT;
  BugReporter &BR;
  
public:
  CheckAttrNonNull(BugReporter &br) : BT(0), BR(br) {}

  virtual bool Audit(ExplodedNode<GRState>* N, GRStateManager& VMgr) {
    CallExpr* CE = cast<CallExpr>(cast<PostStmt>(N->getLocation()).getStmt());
    const GRState* state = N->getState();
    
    SVal X = VMgr.GetSVal(state, CE->getCallee());

    const FunctionDecl* FD = X.getAsFunctionDecl();
    if (!FD)
      return false;

    const NonNullAttr* Att = FD->getAttr<NonNullAttr>();
    
    if (!Att)
      return false;
    
    // Iterate through the arguments of CE and check them for null.
    unsigned idx = 0;
    bool hasError = false;
    
    for (CallExpr::arg_iterator I=CE->arg_begin(), E=CE->arg_end(); I!=E;
         ++I, ++idx) {
      
      if (!VMgr.isEqual(state, *I, 0) || !Att->isNonNull(idx))
        continue;

      // Lazily allocate the BugType object if it hasn't already been created.
      // Ownership is transferred to the BugReporter object once the BugReport
      // is passed to 'EmitWarning'.
      if (!BT) BT =
        new BugType("Argument with 'nonnull' attribute passed null", "API");
      
      RangedBugReport *R = new RangedBugReport(*BT,
                                   "Null pointer passed as an argument to a "
                                   "'nonnull' parameter", N);

      R->addRange((*I)->getSourceRange());
      BR.EmitReport(R);
      hasError = true;
    }
    
    return hasError;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Definitions for bug reporter visitors.
//===----------------------------------------------------------------------===//

namespace {
#if 0
class VISIBILITY_HIDDEN TrackValueBRVisitor : public BugReporterVisitor {
  SVal V;
  Stmt *S;
  const MemRegion *R;
public:
  TrackValueBRVisitor(SVal v, Stmt *s) : V(v), S(s), R(0) {}
  
  PathDiagnosticPiece* VisitNode(const ExplodedNode<GRState> *N,
                                 const ExplodedNode<GRState> *PrevN,
                                 BugReporterContext& BRC) {
    
    // Not at a expression?
    if (!isa<PostStmt>(N->getLocation())) {
      S = 0;
      return NULL;
    }
    
    if (S)
      return VisitNodeExpr(N, PrevN, BRC);
    else if (R)
      return VisitNodeRegion(N, PrevN, BRC);
    
    return NULL;
  }
  
  PathDiagnosticPiece* VisitNodeExpr(const ExplodedNode<GRState> *N,
                                     const ExplodedNode<GRState> *PrevN,
                                     BugReporterContext& BRC) {
    
    assert(S);
    PostStmt P = cast<PostStmt>(N->getLocation());
    Stmt *X = P.getStmt();
    
    // Generate the subexpression path.
    llvm::SmallVector<Stmt*, 4> SubExprPath;
    ParentMap &PM = BRC.getParentMap();
    
    for ( ; X && X != S ; X = X.getParent(X)) {
      if (isa<ParenExpr>(X))
        continue;
      
      SubExprPath.push_back(L);
    }
     
    // Lost track?  (X is not a subexpression of S).
    if (X != S) {
      S = NULL;
      return NULL;
    }

    // Now go down the subexpression path!
    
    
    
  }  
};
#endif
  
class VISIBILITY_HIDDEN TrackConstraintBRVisitor : public BugReporterVisitor {
  SVal Constraint;
  const bool Assumption;
  bool isSatisfied;
public:
  TrackConstraintBRVisitor(SVal constraint, bool assumption)
    : Constraint(constraint), Assumption(assumption), isSatisfied(false) {}
    
  PathDiagnosticPiece* VisitNode(const ExplodedNode<GRState> *N,
                                 const ExplodedNode<GRState> *PrevN,
                                 BugReporterContext& BRC) {
    if (isSatisfied)
      return NULL;
    
    // Check if in the previous state it was feasible for this constraint
    // to *not* be true.
    
    GRStateManager &StateMgr = BRC.getStateManager();
    bool isFeasible = false;    
    if (StateMgr.Assume(PrevN->getState(), Constraint, !Assumption,
                        isFeasible)) {
      assert(isFeasible); // Eventually we don't need 'isFeasible'.

      isSatisfied = true;
      
      // As a sanity check, make sure that the negation of the constraint
      // was infeasible in the current state.  If it is feasible, we somehow
      // missed the transition point.
      isFeasible = false;
      if (StateMgr.Assume(N->getState(), Constraint, !Assumption,
                          isFeasible)) {
        assert(isFeasible);
        return NULL;
      }
      
      // We found the transition point for the constraint.  We now need to
      // pretty-print the constraint. (work-in-progress)      
      std::string sbuf;
      llvm::raw_string_ostream os(sbuf);
      
      if (isa<Loc>(Constraint)) {
        os << "Assuming pointer value is ";
        os << (Assumption ? "non-NULL" : "NULL");
      }
      
      if (os.str().empty())
        return NULL;
      
      // FIXME: Refactor this into BugReporterContext.
      Stmt *S = 0;      
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

static void registerTrackNullValue(BugReporterContext& BRC,
                                   const ExplodedNode<GRState>* N) {
  
  ProgramPoint P = N->getLocation();
  PostStmt *PS = dyn_cast<PostStmt>(&P);

  if (!PS)
    return;
  
  Stmt *S = PS->getStmt();
  
  if (ArraySubscriptExpr *AE = dyn_cast<ArraySubscriptExpr>(S)) {
    S = AE->getBase();
  }
  
  SVal V = BRC.getStateManager().GetSValAsScalarOrLoc(N->getState(), S);
  
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
//      registerTrackValue(BRC, S, V, N);
    }
  }
  
  // Was it a hard integer?
//  if (isa<nonloc::ConcreteInt>(V))
//    registerTrackValue(BRC, S, V, N);  
}

//===----------------------------------------------------------------------===//
// Check registration.
//===----------------------------------------------------------------------===//

void GRExprEngine::RegisterInternalChecks() {
  // Register internal "built-in" BugTypes with the BugReporter. These BugTypes
  // are different than what probably many checks will do since they don't
  // create BugReports on-the-fly but instead wait until GRExprEngine finishes
  // analyzing a function.  Generation of BugReport objects is done via a call
  // to 'FlushReports' from BugReporter.
  BR.Register(new NullDeref(this));
  BR.Register(new UndefinedDeref(this));
  BR.Register(new UndefBranch(this));
  BR.Register(new DivZero(this));
  BR.Register(new UndefResult(this));
  BR.Register(new BadCall(this));
  BR.Register(new RetStack(this));
  BR.Register(new RetUndef(this));
  BR.Register(new BadArg(this));
  BR.Register(new BadMsgExprArg(this));
  BR.Register(new BadReceiver(this));
  BR.Register(new OutOfBoundMemoryAccess(this));
  BR.Register(new BadSizeVLA(this));
  BR.Register(new NilReceiverStructRet(this));
  BR.Register(new NilReceiverLargerThanVoidPtrRet(this));
  
  // The following checks do not need to have their associated BugTypes
  // explicitly registered with the BugReporter.  If they issue any BugReports,
  // their associated BugType will get registered with the BugReporter
  // automatically.  Note that the check itself is owned by the GRExprEngine
  // object.
  AddCheck(new CheckAttrNonNull(BR), Stmt::CallExprClass);
}
