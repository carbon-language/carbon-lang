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
  BR.Register(new UndefResult(this));
  BR.Register(new NilReceiverStructRet(this));
  BR.Register(new NilReceiverLargerThanVoidPtrRet(this));

  // The following checks do not need to have their associated BugTypes
  // explicitly registered with the BugReporter.  If they issue any BugReports,
  // their associated BugType will get registered with the BugReporter
  // automatically.  Note that the check itself is owned by the GRExprEngine
  // object.  
  RegisterAttrNonNullChecker(*this);
  RegisterUndefinedArgChecker(*this);
  RegisterDereferenceChecker(*this);
  RegisterVLASizeChecker(*this);
  RegisterDivZeroChecker(*this);
  RegisterReturnStackAddressChecker(*this);
  RegisterReturnUndefChecker(*this);
  RegisterUndefinedArraySubscriptChecker(*this);
  RegisterUndefinedAssignmentChecker(*this);
  RegisterUndefBranchChecker(*this);
}
