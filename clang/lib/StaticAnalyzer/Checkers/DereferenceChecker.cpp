//== NullDerefChecker.cpp - Null dereference checker ------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines NullDerefChecker, a builtin check in ExprEngine that performs
// checks for null pointers at loads and stores.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"

using namespace clang;
using namespace ento;

namespace {
class DereferenceChecker
    : public Checker< check::Location,
                        EventDispatcher<ImplicitNullDerefEvent> > {
  mutable llvm::OwningPtr<BuiltinBug> BT_null;
  mutable llvm::OwningPtr<BuiltinBug> BT_undef;

public:
  void checkLocation(SVal location, bool isLoad, CheckerContext &C) const;

  static void AddDerefSource(raw_ostream &os,
                             SmallVectorImpl<SourceRange> &Ranges,
                             const Expr *Ex, bool loadedFrom = false);
};
} // end anonymous namespace

void DereferenceChecker::AddDerefSource(raw_ostream &os,
                                     SmallVectorImpl<SourceRange> &Ranges,
                                        const Expr *Ex,
                                        bool loadedFrom) {
  Ex = Ex->IgnoreParenLValueCasts();
  switch (Ex->getStmtClass()) {
    default:
      return;
    case Stmt::DeclRefExprClass: {
      const DeclRefExpr *DR = cast<DeclRefExpr>(Ex);
      if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
        os << " (" << (loadedFrom ? "loaded from" : "from")
           << " variable '" <<  VD->getName() << "')";
        Ranges.push_back(DR->getSourceRange());
      }
      return;
    }
    case Stmt::MemberExprClass: {
      const MemberExpr *ME = cast<MemberExpr>(Ex);
      os << " (" << (loadedFrom ? "loaded from" : "via")
         << " field '" << ME->getMemberNameInfo() << "')";
      SourceLocation L = ME->getMemberLoc();
      Ranges.push_back(SourceRange(L, L));
      break;
    }
  }
}

void DereferenceChecker::checkLocation(SVal l, bool isLoad,
                                       CheckerContext &C) const {
  // Check for dereference of an undefined value.
  if (l.isUndef()) {
    if (ExplodedNode *N = C.generateSink()) {
      if (!BT_undef)
        BT_undef.reset(new BuiltinBug("Dereference of undefined pointer value"));

      EnhancedBugReport *report =
        new EnhancedBugReport(*BT_undef, BT_undef->getDescription(), N);
      report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                                bugreporter::GetDerefExpr(N));
      C.EmitReport(report);
    }
    return;
  }

  DefinedOrUnknownSVal location = cast<DefinedOrUnknownSVal>(l);

  // Check for null dereferences.
  if (!isa<Loc>(location))
    return;

  const Stmt *S = C.getStmt();
  const GRState *state = C.getState();
  const GRState *notNullState, *nullState;
  llvm::tie(notNullState, nullState) = state->assume(location);

  // The explicit NULL case.
  if (nullState) {
    if (!notNullState) {
      // Generate an error node.
      ExplodedNode *N = C.generateSink(nullState);
      if (!N)
        return;

      // We know that 'location' cannot be non-null.  This is what
      // we call an "explicit" null dereference.
      if (!BT_null)
        BT_null.reset(new BuiltinBug("Dereference of null pointer"));

      llvm::SmallString<100> buf;
      SmallVector<SourceRange, 2> Ranges;
      
      // Walk through lvalue casts to get the original expression
      // that syntactically caused the load.
      if (const Expr *expr = dyn_cast<Expr>(S))
        S = expr->IgnoreParenLValueCasts();

      switch (S->getStmtClass()) {
        case Stmt::ArraySubscriptExprClass: {
          llvm::raw_svector_ostream os(buf);
          os << "Array access";
          const ArraySubscriptExpr *AE = cast<ArraySubscriptExpr>(S);
          AddDerefSource(os, Ranges, AE->getBase()->IgnoreParenCasts());
          os << " results in a null pointer dereference";
          break;
        }
        case Stmt::UnaryOperatorClass: {
          llvm::raw_svector_ostream os(buf);
          os << "Dereference of null pointer";
          const UnaryOperator *U = cast<UnaryOperator>(S);
          AddDerefSource(os, Ranges, U->getSubExpr()->IgnoreParens(), true);
          break;
        }
        case Stmt::MemberExprClass: {
          const MemberExpr *M = cast<MemberExpr>(S);
          if (M->isArrow()) {
            llvm::raw_svector_ostream os(buf);
            os << "Access to field '" << M->getMemberNameInfo()
               << "' results in a dereference of a null pointer";
            AddDerefSource(os, Ranges, M->getBase()->IgnoreParenCasts(), true);
          }
          break;
        }
        case Stmt::ObjCIvarRefExprClass: {
          const ObjCIvarRefExpr *IV = cast<ObjCIvarRefExpr>(S);
          if (const DeclRefExpr *DR =
              dyn_cast<DeclRefExpr>(IV->getBase()->IgnoreParenCasts())) {
            if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
              llvm::raw_svector_ostream os(buf);
              os << "Instance variable access (via '" << VD->getName()
                 << "') results in a null pointer dereference";
            }
          }
          Ranges.push_back(IV->getSourceRange());
          break;
        }
        default:
          break;
      }

      EnhancedBugReport *report =
        new EnhancedBugReport(*BT_null,
                              buf.empty() ? BT_null->getDescription():buf.str(),
                              N);

      report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                                bugreporter::GetDerefExpr(N));

      for (SmallVectorImpl<SourceRange>::iterator
            I = Ranges.begin(), E = Ranges.end(); I!=E; ++I)
        report->addRange(*I);

      C.EmitReport(report);
      return;
    }
    else {
      // Otherwise, we have the case where the location could either be
      // null or not-null.  Record the error node as an "implicit" null
      // dereference.
      if (ExplodedNode *N = C.generateSink(nullState)) {
        ImplicitNullDerefEvent event = { l, isLoad, N, &C.getBugReporter() };
        dispatchEvent(event);
      }
    }
  }

  // From this point forward, we know that the location is not null.
  C.addTransition(notNullState);
}

void ento::registerDereferenceChecker(CheckerManager &mgr) {
  mgr.registerChecker<DereferenceChecker>();
}
