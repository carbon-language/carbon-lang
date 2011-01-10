//== CheckerVisitor.h - Abstract visitor for checkers ------------*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines CheckerVisitor.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_CHECKERVISITOR
#define LLVM_CLANG_GR_CHECKERVISITOR
#include "clang/StaticAnalyzer/PathSensitive/Checker.h"

namespace clang {

namespace ento {

//===----------------------------------------------------------------------===//
// Checker visitor interface.  Used by subclasses of Checker to specify their
// own checker visitor logic.
//===----------------------------------------------------------------------===//

/// CheckerVisitor - This class implements a simple visitor for Stmt subclasses.
/// Since Expr derives from Stmt, this also includes support for visiting Exprs.
template<typename ImplClass>
class CheckerVisitor : public Checker {
public:
  virtual void _PreVisit(CheckerContext &C, const Stmt *S) {
    PreVisit(C, S);
  }
  
  virtual void _PostVisit(CheckerContext &C, const Stmt *S) {
    PostVisit(C, S);
  }

  void PreVisit(CheckerContext &C, const Stmt *S) {
    switch (S->getStmtClass()) {
      default:
        assert(false && "Unsupport statement.");
        return;

#define PREVISIT(NAME, FALLBACK) \
case Stmt::NAME ## Class:\
static_cast<ImplClass*>(this)->PreVisit ## NAME(C,static_cast<const NAME*>(S));\
break;
#include "clang/StaticAnalyzer/PathSensitive/CheckerVisitor.def"
    }
  }
  
  void PostVisit(CheckerContext &C, const Stmt *S) {
    switch (S->getStmtClass()) {
      default:
        assert(false && "Unsupport statement.");
        return;

#define POSTVISIT(NAME, FALLBACK) \
case Stmt::NAME ## Class:\
static_cast<ImplClass*>(this)->\
PostVisit ## NAME(C,static_cast<const NAME*>(S));\
break;
#include "clang/StaticAnalyzer/PathSensitive/CheckerVisitor.def"
    }
  }

  void PreVisitGenericCall(CheckerContext &C, const CallExpr *CE) {
    static_cast<ImplClass*>(this)->PreVisitStmt(C, CE);
  }
  void PostVisitGenericCall(CheckerContext &C, const CallExpr *CE) {
    static_cast<ImplClass*>(this)->PostVisitStmt(C, CE);
  }

  void PreVisitStmt(CheckerContext &C, const Stmt *S) {
    *C.respondsToCallback = false;
  }

  void PostVisitStmt(CheckerContext &C, const Stmt *S) {
    *C.respondsToCallback = false;
  }

  void PreVisitCastExpr(CheckerContext &C, const CastExpr *E) {
    static_cast<ImplClass*>(this)->PreVisitStmt(C, E);
  }
  
#define PREVISIT(NAME, FALLBACK) \
void PreVisit ## NAME(CheckerContext &C, const NAME* S) {\
  static_cast<ImplClass*>(this)->PreVisit ## FALLBACK(C, S);\
}
#define POSTVISIT(NAME, FALLBACK) \
void PostVisit ## NAME(CheckerContext &C, const NAME* S) {\
  static_cast<ImplClass*>(this)->PostVisit ## FALLBACK(C, S);\
}
#include "clang/StaticAnalyzer/PathSensitive/CheckerVisitor.def"
};

} // end GR namespace

} // end clang namespace

#endif
