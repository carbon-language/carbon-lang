//== NullDerefChecker.cpp - Null dereference checker ------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines NullDerefChecker, a builtin check in GRExprEngine that performs
// checks for null pointers at loads and stores.
//
//===----------------------------------------------------------------------===//

#include "GRExprEngineInternalChecks.h"
#include "clang/Checker/BugReporter/BugType.h"
#include "clang/Checker/Checkers/DereferenceChecker.h"
#include "clang/Checker/PathSensitive/Checker.h"
#include "clang/Checker/PathSensitive/GRExprEngine.h"

using namespace clang;

namespace {
class DereferenceChecker : public Checker {
  BuiltinBug *BT_null;
  BuiltinBug *BT_undef;
  llvm::SmallVector<ExplodedNode*, 2> ImplicitNullDerefNodes;
public:
  DereferenceChecker() : BT_null(0), BT_undef(0) {}
  static void *getTag() { static int tag = 0; return &tag; }
  void visitLocation(CheckerContext &C, const Stmt *S, SVal location);

  std::pair<ExplodedNode * const*, ExplodedNode * const*>
  getImplicitNodes() const {
    return std::make_pair(ImplicitNullDerefNodes.data(),
                          ImplicitNullDerefNodes.data() +
                          ImplicitNullDerefNodes.size());
  }
  void AddDerefSource(llvm::raw_ostream &os,
                      llvm::SmallVectorImpl<SourceRange> &Ranges,
                      const Expr *Ex, bool loadedFrom = false);
};
} // end anonymous namespace

void clang::RegisterDereferenceChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new DereferenceChecker());
}

std::pair<ExplodedNode * const *, ExplodedNode * const *>
clang::GetImplicitNullDereferences(GRExprEngine &Eng) {
  DereferenceChecker *checker = Eng.getChecker<DereferenceChecker>();
  if (!checker)
    return std::make_pair((ExplodedNode * const *) 0,
                          (ExplodedNode * const *) 0);
  return checker->getImplicitNodes();
}

void DereferenceChecker::AddDerefSource(llvm::raw_ostream &os,
                                     llvm::SmallVectorImpl<SourceRange> &Ranges,
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

void DereferenceChecker::visitLocation(CheckerContext &C, const Stmt *S,
                                       SVal l) {
  // Check for dereference of an undefined value.
  if (l.isUndef()) {
    if (ExplodedNode *N = C.generateSink()) {
      if (!BT_undef)
        BT_undef = new BuiltinBug("Dereference of undefined pointer value");

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
        BT_null = new BuiltinBug("Dereference of null pointer");

      llvm::SmallString<100> buf;
      llvm::SmallVector<SourceRange, 2> Ranges;
      
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

      for (llvm::SmallVectorImpl<SourceRange>::iterator
            I = Ranges.begin(), E = Ranges.end(); I!=E; ++I)
        report->addRange(*I);

      C.EmitReport(report);
      return;
    }
    else {
      // Otherwise, we have the case where the location could either be
      // null or not-null.  Record the error node as an "implicit" null
      // dereference.
      if (ExplodedNode *N = C.generateSink(nullState))
        ImplicitNullDerefNodes.push_back(N);
    }
  }

  // From this point forward, we know that the location is not null.
  C.addTransition(notNullState);
}
