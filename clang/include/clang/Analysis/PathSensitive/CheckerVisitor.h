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
#include "clang/Analysis/PathSensitive/Checker.h"

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
  virtual void _PreVisit(CheckerContext &C, const Stmt *stmt) {
    PreVisit(C, stmt);
  }

  void PreVisit(CheckerContext &C, const Stmt *S) {
    switch (S->getStmtClass()) {
      default:
        assert(false && "Unsupport statement.");
        return;

      case Stmt::ImplicitCastExprClass:
      case Stmt::ExplicitCastExprClass:
      case Stmt::CStyleCastExprClass:
        static_cast<ImplClass*>(this)->PreVisitCastExpr(C,
                                               static_cast<const CastExpr*>(S));
        break;

      case Stmt::CompoundAssignOperatorClass:
        static_cast<ImplClass*>(this)->PreVisitBinaryOperator(C,
                                         static_cast<const BinaryOperator*>(S));
        break;

#define PREVISIT(NAME) \
case Stmt::NAME ## Class:\
static_cast<ImplClass*>(this)->PreVisit ## NAME(C,static_cast<const NAME*>(S));\
break;
#include "clang/Analysis/PathSensitive/CheckerVisitor.def"
    }
  }

#define PREVISIT(NAME) \
void PreVisit ## NAME(CheckerContext &C, const NAME* S) {}
#include "clang/Analysis/PathSensitive/CheckerVisitor.def"
};

} // end clang namespace

#endif

