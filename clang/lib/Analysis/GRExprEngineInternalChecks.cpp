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
// Bug Descriptions.
//===----------------------------------------------------------------------===//

namespace {
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
  
  template <typename ITER>
  void Emit(BugReporter& BR, ITER I, ITER E) {
    for (; I != E; ++I) BR.EmitReport(new BugReport(*this, desc.c_str(),
                                                     GetNode(I)));
  }  
};
  
class VISIBILITY_HIDDEN NullDeref : public BuiltinBug {
public:
  NullDeref(GRExprEngine* eng)
    : BuiltinBug(eng,"null dereference", "Dereference of null pointer.") {}

  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    Emit(BR, Eng.null_derefs_begin(), Eng.null_derefs_end());
  }
};
  
class VISIBILITY_HIDDEN UndefinedDeref : public BuiltinBug {
public:
  UndefinedDeref(GRExprEngine* eng)
    : BuiltinBug(eng,"uninitialized pointer dereference",
                 "Dereference of undefined value.") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    Emit(BR, Eng.undef_derefs_begin(), Eng.undef_derefs_end());
  }
};

class VISIBILITY_HIDDEN DivZero : public BuiltinBug {
public:
  DivZero(GRExprEngine* eng)
    : BuiltinBug(eng,"divide-by-zero", "Division by zero/undefined value.") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    Emit(BR, Eng.explicit_bad_divides_begin(), Eng.explicit_bad_divides_end());
  }
};
  
class VISIBILITY_HIDDEN UndefResult : public BuiltinBug {
public:
  UndefResult(GRExprEngine* eng) : BuiltinBug(eng,"undefined result",
                             "Result of operation is undefined.") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    Emit(BR, Eng.undef_results_begin(), Eng.undef_results_end());
  }
};
  
class VISIBILITY_HIDDEN BadCall : public BuiltinBug {
public:
  BadCall(GRExprEngine *eng)
  : BuiltinBug(eng,"invalid function call",
        "Called function is a NULL or an undefined function pointer value.") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    Emit(BR, Eng.bad_calls_begin(), Eng.bad_calls_end());
  }
};

class VISIBILITY_HIDDEN BadArg : public BuiltinBug {
public:
  BadArg(GRExprEngine* eng) : BuiltinBug(eng,"uninitialized argument",  
    "Pass-by-value argument in function is undefined.") {}

  BadArg(GRExprEngine* eng, const char* d)
    : BuiltinBug(eng,"uninitialized argument", d) {}
  
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
    : BadArg(eng,"Pass-by-value argument in message expression is undefined."){}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    for (GRExprEngine::UndefArgsTy::iterator I=Eng.msg_expr_undef_arg_begin(),
         E = Eng.msg_expr_undef_arg_end(); I!=E; ++I) {      
      // Generate a report for this bug.
      RangedBugReport *report = new RangedBugReport(*this, desc.c_str(), I->first);
      report->addRange(I->second->getSourceRange());
      BR.EmitReport(report);
    }    
  }
};
  
class VISIBILITY_HIDDEN BadReceiver : public BuiltinBug {
public:  
  BadReceiver(GRExprEngine* eng)
  : BuiltinBug(eng,"uninitialized receiver",
               "Receiver in message expression is an uninitialized value.") {}
  
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
  RetStack(GRExprEngine* eng) : BuiltinBug(eng, "return of stack address") {}
  
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
  RetUndef(GRExprEngine* eng) : BuiltinBug(eng,"uninitialized return value",
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
    : BuiltinBug(eng,"uninitialized value",
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
    : BuiltinBug(eng,"out-of-bound memory access",
                     "Load or store into an out-of-bound memory position.") {}

  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    Emit(BR, Eng.explicit_oob_memacc_begin(), Eng.explicit_oob_memacc_end());
  }
};
  
class VISIBILITY_HIDDEN BadSizeVLA : public BuiltinBug {
public:
  BadSizeVLA(GRExprEngine* eng) : BuiltinBug(eng, "bad VLA size") {}
  
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
      os << "The expression used to specify the number of elements in the VLA '"
         << VD->getNameAsString() << "' evaluates to ";
      
      if (Eng.getStateManager().GetSVal(N->getState(), SizeExpr).isUndef())
        os << "an undefined or garbage value.";
      else
        os << "0. VLAs with no elements have undefined behavior.";

      RangedBugReport *report = new RangedBugReport(*this, os.str().c_str(), N);
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
    
    if (!isa<loc::FuncVal>(X))
      return false;
    
    FunctionDecl* FD = dyn_cast<FunctionDecl>(cast<loc::FuncVal>(X).getDecl());
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
      if (!BT) BT = new BugType("'nonnull' argument passed null", "API");
      
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
  
  // The following checks do not need to have their associated BugTypes
  // explicitly registered with the BugReporter.  If they issue any BugReports,
  // their associated BugType will get registered with the BugReporter
  // automatically.  Note that the check itself is owned by the GRExprEngine
  // object.
  AddCheck(new CheckAttrNonNull(BR), Stmt::CallExprClass);
}
