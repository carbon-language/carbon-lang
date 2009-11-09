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

#include "GRExprEngineInternalChecks.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Analysis/PathSensitive/CheckerVisitor.h"
#include "clang/Analysis/PathSensitive/Checkers/DereferenceChecker.h"
#include "clang/Analysis/PathSensitive/Checkers/BadCallChecker.h"
#include "clang/Analysis/PathSensitive/Checkers/UndefinedArgChecker.h"
#include "clang/Analysis/PathSensitive/Checkers/UndefinedAssignmentChecker.h"
#include "clang/Analysis/PathSensitive/Checkers/AttrNonNullChecker.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::bugreporter;

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

template <typename ITERATOR> inline
ExplodedNode* GetNode(ITERATOR I) {
  return *I;
}

template <> inline
ExplodedNode* GetNode(GRExprEngine::undef_arg_iterator I) {
  return I->first;
}

//===----------------------------------------------------------------------===//
// Bug Descriptions.
//===----------------------------------------------------------------------===//
namespace clang {
class BuiltinBugReport : public RangedBugReport {
public:
  BuiltinBugReport(BugType& bt, const char* desc,
                   ExplodedNode *n)
  : RangedBugReport(bt, desc, n) {}

  BuiltinBugReport(BugType& bt, const char *shortDesc, const char *desc,
                   ExplodedNode *n)
  : RangedBugReport(bt, shortDesc, desc, n) {}

  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode* N);
};

void BuiltinBugReport::registerInitialVisitors(BugReporterContext& BRC,
                                               const ExplodedNode* N) {
  static_cast<BuiltinBug&>(getBugType()).registerInitialVisitors(BRC, N, this);
}

template <typename ITER>
void BuiltinBug::Emit(BugReporter& BR, ITER I, ITER E) {
  for (; I != E; ++I) BR.EmitReport(new BuiltinBugReport(*this, desc.c_str(),
                                                         GetNode(I)));
}

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
      const ObjCMessageExpr *ME = cast<ObjCMessageExpr>(P.getStmt());
      os << "The receiver in the message expression is 'nil' and results in the"
            " returned value (of type '"
         << ME->getType().getAsString()
         << "') to be garbage or otherwise undefined";

      BuiltinBugReport *R = new BuiltinBugReport(*this, os.str().c_str(), *I);
      R->addRange(ME->getReceiver()->getSourceRange());
      BR.EmitReport(R);
    }
  }

  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode* N,
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
      const ObjCMessageExpr *ME = cast<ObjCMessageExpr>(P.getStmt());
      os << "The receiver in the message expression is 'nil' and results in the"
      " returned value (of type '"
      << ME->getType().getAsString()
      << "' and of size "
      << Eng.getContext().getTypeSize(ME->getType()) / 8
      << " bytes) to be garbage or otherwise undefined";

      BuiltinBugReport *R = new BuiltinBugReport(*this, os.str().c_str(), *I);
      R->addRange(ME->getReceiver()->getSourceRange());
      BR.EmitReport(R);
    }
  }
  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode* N,
                               BuiltinBugReport *R) {
    registerTrackNullOrUndefValue(BRC, GetReceiverExpr(N), N);
  }
};

class VISIBILITY_HIDDEN UndefResult : public BuiltinBug {
public:
  UndefResult(GRExprEngine* eng)
    : BuiltinBug(eng,"Undefined or garbage result",
                 "Result of operation is garbage or undefined") {}

  void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {
    for (GRExprEngine::undef_result_iterator I=Eng.undef_results_begin(),
         E = Eng.undef_results_end(); I!=E; ++I) {
      
      ExplodedNode *N = *I;
      const Stmt *S = N->getLocationAs<PostStmt>()->getStmt();        
      BuiltinBugReport *report = NULL;
      
      if (const BinaryOperator *B = dyn_cast<BinaryOperator>(S)) {
        llvm::SmallString<256> sbuf;
        llvm::raw_svector_ostream OS(sbuf);
        const GRState *ST = N->getState();
        const Expr *Ex = NULL;
        bool isLeft = true;

        if (ST->getSVal(B->getLHS()).isUndef()) {
          Ex = B->getLHS()->IgnoreParenCasts();
          isLeft = true;
        }
        else if (ST->getSVal(B->getRHS()).isUndef()) {
          Ex = B->getRHS()->IgnoreParenCasts();
          isLeft = false;
        }
                
        if (Ex) {
          OS << "The " << (isLeft ? "left" : "right")
             << " operand of '"
             << BinaryOperator::getOpcodeStr(B->getOpcode())
             << "' is a garbage value";
        }          
        else {
          // Neither operand was undefined, but the result is undefined.
          OS << "The result of the '"
             << BinaryOperator::getOpcodeStr(B->getOpcode())
             << "' expression is undefined";
        }
      
        // FIXME: Use StringRefs to pass string information.
        report = new BuiltinBugReport(*this, OS.str().str().c_str(), N);
        if (Ex) report->addRange(Ex->getSourceRange());
      }
      else {
        report = new BuiltinBugReport(*this, 
                                      "Expression evaluates to an uninitialized"
                                      " or undefined value", N);
      }

      BR.EmitReport(report);
    }
  }
  
  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode* N,
                               BuiltinBugReport *R) {
    
    const Stmt *S = N->getLocationAs<StmtPoint>()->getStmt();
    const Stmt *X = S;
    
    if (const BinaryOperator *B = dyn_cast<BinaryOperator>(S)) {
      const GRState *ST = N->getState();
      if (ST->getSVal(B->getLHS()).isUndef())
        X = B->getLHS();
      else if (ST->getSVal(B->getRHS()).isUndef())
        X = B->getRHS();
    }
    
    registerTrackNullOrUndefValue(BRC, X, N);
  }
};

class VISIBILITY_HIDDEN ArgReport : public BuiltinBugReport {
  const Stmt *Arg;
public:
  ArgReport(BugType& bt, const char* desc, ExplodedNode *n,
         const Stmt *arg)
  : BuiltinBugReport(bt, desc, n), Arg(arg) {}

  ArgReport(BugType& bt, const char *shortDesc, const char *desc,
                   ExplodedNode *n, const Stmt *arg)
  : BuiltinBugReport(bt, shortDesc, desc, n), Arg(arg) {}

  const Stmt *getArg() const { return Arg; }
};

class VISIBILITY_HIDDEN BadArg : public BuiltinBug {
public:
  BadArg(GRExprEngine* eng=0) : BuiltinBug(eng,"Uninitialized argument",
    "Pass-by-value argument in function call is undefined") {}

  BadArg(GRExprEngine* eng, const char* d)
    : BuiltinBug(eng,"Uninitialized argument", d) {}

  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode* N,
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
      ExplodedNode* N = *I;
      const Stmt *S = cast<PostStmt>(N->getLocation()).getStmt();
      const Expr* E = cast<ObjCMessageExpr>(S)->getReceiver();
      assert (E && "Receiver cannot be NULL");
      report->addRange(E->getSourceRange());
      BR.EmitReport(report);
    }
  }

  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode* N,
                               BuiltinBugReport *R) {
    registerTrackNullOrUndefValue(BRC, GetReceiverExpr(N), N);
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

    bool MatchesCriteria(Expr* Ex) { return St->getSVal(Ex).isUndef(); }
  };

public:
  UndefBranch(GRExprEngine *eng)
    : BuiltinBug(eng,"Use of garbage value",
                 "Branch condition evaluates to an undefined or garbage value")
       {}

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
      ExplodedNode *N = *(*I)->pred_begin();
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
                               const ExplodedNode* N,
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

} // end clang namespace

//===----------------------------------------------------------------------===//
// Check registration.
//===----------------------------------------------------------------------===//

void GRExprEngine::RegisterInternalChecks() {
  // Register internal "built-in" BugTypes with the BugReporter. These BugTypes
  // are different than what probably many checks will do since they don't
  // create BugReports on-the-fly but instead wait until GRExprEngine finishes
  // analyzing a function.  Generation of BugReport objects is done via a call
  // to 'FlushReports' from BugReporter.
  BR.Register(new UndefBranch(this));
  BR.Register(new UndefResult(this));
  BR.Register(new BadMsgExprArg(this));
  BR.Register(new BadReceiver(this));
  BR.Register(new OutOfBoundMemoryAccess(this));
  BR.Register(new NilReceiverStructRet(this));
  BR.Register(new NilReceiverLargerThanVoidPtrRet(this));

  // The following checks do not need to have their associated BugTypes
  // explicitly registered with the BugReporter.  If they issue any BugReports,
  // their associated BugType will get registered with the BugReporter
  // automatically.  Note that the check itself is owned by the GRExprEngine
  // object.  
  registerCheck(new AttrNonNullChecker());
  registerCheck(new UndefinedArgChecker());
  registerCheck(new UndefinedAssignmentChecker());
  registerCheck(new BadCallChecker());
  registerCheck(new UndefDerefChecker());
  registerCheck(new NullDerefChecker());
  
  RegisterVLASizeChecker(*this);
  RegisterDivZeroChecker(*this);
  RegisterReturnStackAddressChecker(*this);
  RegisterReturnUndefChecker(*this);
  RegisterPointerSubChecker(*this);
  RegisterFixedAddressChecker(*this);
  // Note that this must be registered after ReturnStackAddressChecker.
  RegisterReturnPointerRangeChecker(*this);

  RegisterCastToStructChecker(*this);
}
