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

static const Stmt *GetDerefExpr(const ExplodedNode<GRState> *N);
static const Stmt *GetReceiverExpr(const ExplodedNode<GRState> *N);
static const Stmt *GetDenomExpr(const ExplodedNode<GRState> *N);
static const Stmt *GetCalleeExpr(const ExplodedNode<GRState> *N);
static const Stmt *GetRetValExpr(const ExplodedNode<GRState> *N);

static void registerTrackNullOrUndefValue(BugReporterContext& BRC,
                                          const Stmt *ValExpr,
                                          const ExplodedNode<GRState>* N);

//===----------------------------------------------------------------------===//
// Bug Descriptions.
//===----------------------------------------------------------------------===//

namespace {

class VISIBILITY_HIDDEN BuiltinBugReport : public RangedBugReport {
public:
  BuiltinBugReport(BugType& bt, const char* desc,
                   ExplodedNode<GRState> *n)
  : RangedBugReport(bt, desc, n) {}
  
  BuiltinBugReport(BugType& bt, const char *shortDesc, const char *desc,
                   ExplodedNode<GRState> *n)
  : RangedBugReport(bt, shortDesc, desc, n) {}  
  
  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode<GRState>* N);
};  
  
class VISIBILITY_HIDDEN BuiltinBug : public BugType {
  GRExprEngine &Eng;
protected:
  const std::string desc;
public:
  BuiltinBug(GRExprEngine *eng, const char* n, const char* d)
    : BugType(n, "Logic errors"), Eng(*eng), desc(d) {}

  BuiltinBug(GRExprEngine *eng, const char* n)
    : BugType(n, "Logic errors"), Eng(*eng), desc(n) {}
  
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
    registerTrackNullOrUndefValue(BRC, GetDerefExpr(N), N);
  }
};
  
class VISIBILITY_HIDDEN NilReceiverStructRet : public BuiltinBug {
public:
  NilReceiverStructRet(GRExprEngine* eng) :
    BuiltinBug(eng, "'nil' receiver with struct return type") {}

  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
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

      BuiltinBugReport *R = new BuiltinBugReport(*this, os.str().c_str(), *I);
      R->addRange(ME->getReceiver()->getSourceRange());
      BR.EmitReport(R);
    }
  }
  
  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode<GRState>* N,
                               BuiltinBugReport *R) {
    registerTrackNullOrUndefValue(BRC, GetReceiverExpr(N), N);
  }
};

class VISIBILITY_HIDDEN NilReceiverLargerThanVoidPtrRet : public BuiltinBug {
public:
  NilReceiverLargerThanVoidPtrRet(GRExprEngine* eng) :
    BuiltinBug(eng,
               "'nil' receiver with return type larger than sizeof(void *)") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
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
      
      BuiltinBugReport *R = new BuiltinBugReport(*this, os.str().c_str(), *I);
      R->addRange(ME->getReceiver()->getSourceRange());
      BR.EmitReport(R);
    }
  }    
  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode<GRState>* N,
                               BuiltinBugReport *R) {
    registerTrackNullOrUndefValue(BRC, GetReceiverExpr(N), N);
  }
};
  
class VISIBILITY_HIDDEN UndefinedDeref : public BuiltinBug {
public:
  UndefinedDeref(GRExprEngine* eng)
    : BuiltinBug(eng,"Dereference of undefined pointer value") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    Emit(BR, Eng.undef_derefs_begin(), Eng.undef_derefs_end());
  }
  
  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode<GRState>* N,
                               BuiltinBugReport *R) {
    registerTrackNullOrUndefValue(BRC, GetDerefExpr(N), N);
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
  
  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode<GRState>* N,
                               BuiltinBugReport *R) {
    registerTrackNullOrUndefValue(BRC, GetDenomExpr(N), N);
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
  
  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode<GRState>* N,
                               BuiltinBugReport *R) {
    registerTrackNullOrUndefValue(BRC, GetCalleeExpr(N), N);
  }
};


class VISIBILITY_HIDDEN ArgReport : public BuiltinBugReport {
  const Stmt *Arg;
public:
  ArgReport(BugType& bt, const char* desc, ExplodedNode<GRState> *n,
         const Stmt *arg)
  : BuiltinBugReport(bt, desc, n), Arg(arg) {}
  
  ArgReport(BugType& bt, const char *shortDesc, const char *desc,
                   ExplodedNode<GRState> *n, const Stmt *arg)
  : BuiltinBugReport(bt, shortDesc, desc, n), Arg(arg) {}  
  
  const Stmt *getArg() const { return Arg; }    
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
      ArgReport *report = new ArgReport(*this, desc.c_str(), I->first,
                                        I->second);
      report->addRange(I->second->getSourceRange());
      BR.EmitReport(report);
    }
  }

  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode<GRState>* N,
                               BuiltinBugReport *R) {
    registerTrackNullOrUndefValue(BRC, static_cast<ArgReport*>(R)->getArg(),
                                  N);
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
      ArgReport *report = new ArgReport(*this, desc.c_str(), I->first,
                                        I->second);
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
      BuiltinBugReport *report = new BuiltinBugReport(*this, desc.c_str(), *I);
      ExplodedNode<GRState>* N = *I;
      Stmt *S = cast<PostStmt>(N->getLocation()).getStmt();
      Expr* E = cast<ObjCMessageExpr>(S)->getReceiver();
      assert (E && "Receiver cannot be NULL");
      report->addRange(E->getSourceRange());
      BR.EmitReport(report);
    }
  }

  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode<GRState>* N,
                               BuiltinBugReport *R) {
    registerTrackNullOrUndefValue(BRC, GetReceiverExpr(N), N);
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
              "Uninitialized or undefined value returned to caller.") {}
  
  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    Emit(BR, Eng.ret_undef_begin(), Eng.ret_undef_end());
  }
  
  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode<GRState>* N,
                               BuiltinBugReport *R) {
    registerTrackNullOrUndefValue(BRC, GetRetValExpr(N), N);
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

      ArgReport *R = new ArgReport(*this, desc.c_str(), *I, Ex);
      R->addRange(Ex->getSourceRange());
      BR.EmitReport(R);
    }
  }
  
  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode<GRState>* N,
                               BuiltinBugReport *R) {
    registerTrackNullOrUndefValue(BRC, static_cast<ArgReport*>(R)->getArg(),
                                  N);
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

      ArgReport *report = new ArgReport(*this, os_short.str().c_str(),
                                        os.str().c_str(), N, SizeExpr);

      report->addRange(SizeExpr->getSourceRange());
      BR.EmitReport(report);
    }
  }
  
  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode<GRState>* N,
                               BuiltinBugReport *R) {
    registerTrackNullOrUndefValue(BRC, static_cast<ArgReport*>(R)->getArg(),
                                  N);
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

static const Stmt *GetDerefExpr(const ExplodedNode<GRState> *N) {
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

static const Stmt *GetReceiverExpr(const ExplodedNode<GRState> *N) {
  const Stmt *S = N->getLocationAs<PostStmt>()->getStmt();
  if (const ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(S))
    return ME->getReceiver();
  return NULL;
}
  
static const Stmt *GetDenomExpr(const ExplodedNode<GRState> *N) {
  const Stmt *S = N->getLocationAs<PostStmt>()->getStmt();
  if (const BinaryOperator *BE = dyn_cast<BinaryOperator>(S))
    return BE->getRHS();
  return NULL;
}
  
static const Stmt *GetCalleeExpr(const ExplodedNode<GRState> *N) {
  const Stmt *S = N->getLocationAs<PostStmt>()->getStmt();
  if (const CallExpr *CE = dyn_cast<CallExpr>(S))
    return CE->getCallee();
  return NULL;
}
  
static const Stmt *GetRetValExpr(const ExplodedNode<GRState> *N) {
  const Stmt *S = N->getLocationAs<PostStmt>()->getStmt();
  if (const ReturnStmt *RS = dyn_cast<ReturnStmt>(S))
    return RS->getRetValue();
  return NULL;
}

namespace {
class VISIBILITY_HIDDEN FindLastStoreBRVisitor : public BugReporterVisitor {
  const MemRegion *R;
  SVal V;
  bool satisfied;
  const ExplodedNode<GRState> *StoreSite;
public:
  FindLastStoreBRVisitor(SVal v, const MemRegion *r)
    : R(r), V(v), satisfied(false), StoreSite(0) {}
                         
  PathDiagnosticPiece* VisitNode(const ExplodedNode<GRState> *N,
                                 const ExplodedNode<GRState> *PrevN,
                                 BugReporterContext& BRC) {
        
    if (satisfied)
      return NULL;

    if (!StoreSite) {      
      GRStateManager &StateMgr = BRC.getStateManager();
      const ExplodedNode<GRState> *Node = N, *Last = NULL;

      for ( ; Node ; Last = Node, Node = Node->getFirstPred()) {
        
        if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
          if (const PostStmt *P = Node->getLocationAs<PostStmt>())
            if (const DeclStmt *DS = P->getStmtAs<DeclStmt>())
              if (DS->getSingleDecl() == VR->getDecl()) {
                Last = Node;
                break;
              }
        }
        
        if (StateMgr.GetSVal(Node->getState(), R) != V)
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
          if (R->isBoundable(C)) {
            if (const TypedRegion *TR = dyn_cast<TypedRegion>(R)) {
              if (C.isObjCObjectPointerType(TR->getValueType(C))) {
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
        if (R->isBoundable(C)) {
          if (const TypedRegion *TR = dyn_cast<TypedRegion>(R)) {
            if (C.isObjCObjectPointerType(TR->getValueType(C))) {
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
      else
        return NULL;
    
      if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
        os << '\'' << VR->getDecl()->getNameAsString() << '\'';
      }
      else
        return NULL;
    }
      
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
        os << (Assumption ? "non-null" : "null");
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

static void registerTrackNullOrUndefValue(BugReporterContext& BRC,
                                          const Stmt *S,
                                          const ExplodedNode<GRState>* N) {
  
  if (!S)
    return;

  GRStateManager &StateMgr = BRC.getStateManager();
  const GRState *state = N->getState();  
  
  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(S)) {        
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {                
      const VarRegion *R =
        StateMgr.getRegionManager().getVarRegion(VD);

      // What did we load?
      SVal V = StateMgr.GetSVal(state, S);
        
      if (isa<loc::ConcreteInt>(V) || isa<nonloc::ConcreteInt>(V) 
          || V.isUndef()) {
        registerFindLastStore(BRC, R, V);
      }
    }
  }
    
  SVal V = StateMgr.GetSValAsScalarOrLoc(state, S);
  
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
