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

#ifndef LLVM_CLANG_ANALYSIS_CHECKERVISITOR
#define LLVM_CLANG_ANALYSIS_CHECKERVISITOR
#include "clang/Checker/PathSensitive/Checker.h"

namespace clang {

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

      case Stmt::ImplicitCastExprClass:
      case Stmt::CStyleCastExprClass:
        static_cast<ImplClass*>(this)->PreVisitCastExpr(C,
                                               static_cast<const CastExpr*>(S));
        break;

      case Stmt::CompoundAssignOperatorClass:
        static_cast<ImplClass*>(this)->PreVisitBinaryOperator(C,
                                         static_cast<const BinaryOperator*>(S));
        break;

#define PREVISIT(NAME, FALLBACK) \
case Stmt::NAME ## Class:\
static_cast<ImplClass*>(this)->PreVisit ## NAME(C,static_cast<const NAME*>(S));\
break;
#include "clang/Checker/PathSensitive/CheckerVisitor.def"
    }
  }
  
  void PostVisit(CheckerContext &C, const Stmt *S) {
    switch (S->getStmtClass()) {
      default:
        assert(false && "Unsupport statement.");
        return;
      case Stmt::CompoundAssignOperatorClass:
        static_cast<ImplClass*>(this)->PostVisitBinaryOperator(C,
                                         static_cast<const BinaryOperator*>(S));
        break;

#define POSTVISIT(NAME, FALLBACK) \
case Stmt::NAME ## Class:\
static_cast<ImplClass*>(this)->\
PostVisit ## NAME(C,static_cast<const NAME*>(S));\
break;
#include "clang/Checker/PathSensitive/CheckerVisitor.def"
    }
  }

  void PreVisitStmt(CheckerContext &C, const Stmt *S) {}
  void PostVisitStmt(CheckerContext &C, const Stmt *S) {}

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
#include "clang/Checker/PathSensitive/CheckerVisitor.def"
};

} // end clang namespace

#endif
