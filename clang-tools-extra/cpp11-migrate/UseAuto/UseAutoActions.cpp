//===-- UseAuto/UseAutoActions.cpp - Matcher callback impl ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  \brief This file contains the implementation of callbacks for the UseAuto
///  transform.
///
//===----------------------------------------------------------------------===//
#include "UseAutoActions.h"
#include "UseAutoMatchers.h"
#include "clang/AST/ASTContext.h"

using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace clang;

void IteratorReplacer::run(const MatchFinder::MatchResult &Result) {
  const DeclStmt *D = Result.Nodes.getNodeAs<DeclStmt>(IteratorDeclStmtId);
  assert(D && "Bad Callback. No node provided");

  SourceManager &SM = *Result.SourceManager;
  if (!Owner.isFileModifiable(SM, D->getLocStart()))
    return;

  for (clang::DeclStmt::const_decl_iterator DI = D->decl_begin(),
                                            DE = D->decl_end();
       DI != DE; ++DI) {
    const VarDecl *V = cast<VarDecl>(*DI);

    const Expr *ExprInit = V->getInit();

    // Skip expressions with cleanups from the initializer expression.
    if (const ExprWithCleanups *E = dyn_cast<ExprWithCleanups>(ExprInit))
      ExprInit = E->getSubExpr();

    const CXXConstructExpr *Construct = cast<CXXConstructExpr>(ExprInit);

    assert(Construct->getNumArgs() == 1u &&
           "Expected constructor with single argument");

    // Drill down to the as-written initializer.
    const Expr *E = Construct->arg_begin()->IgnoreParenImpCasts();
    if (E != E->IgnoreConversionOperator())
      // We hit a conversion operator. Early-out now as they imply an implicit
      // conversion from a different type. Could also mean an explicit
      // conversion from the same type but that's pretty rare.
      return;

    if (const CXXConstructExpr *NestedConstruct = dyn_cast<CXXConstructExpr>(E))
      // If we ran into an implicit conversion constructor, can't convert.
      //
      // FIXME: The following only checks if the constructor can be used
      // implicitly, not if it actually was. Cases where the converting
      // constructor was used explicitly won't get converted.
      if (NestedConstruct->getConstructor()->isConvertingConstructor(false))
        return;
    if (!Result.Context->hasSameType(V->getType(), E->getType()))
      return;
  }
  // Get the type location using the first declartion.
  const VarDecl *V = cast<VarDecl>(*D->decl_begin());
  TypeLoc TL = V->getTypeSourceInfo()->getTypeLoc();

  // WARNING: TypeLoc::getSourceRange() will include the identifier for things
  // like function pointers. Not a concern since this action only works with
  // iterators but something to keep in mind in the future.

  CharSourceRange Range(TL.getSourceRange(), true);
  Replace.insert(tooling::Replacement(SM, Range, "auto"));
  ++AcceptedChanges;
}

void NewReplacer::run(const MatchFinder::MatchResult &Result) {
  const DeclStmt *D = Result.Nodes.getNodeAs<DeclStmt>(DeclWithNewId);
  assert(D && "Bad Callback. No node provided");

  SourceManager &SM = *Result.SourceManager;
  if (!Owner.isFileModifiable(SM, D->getLocStart()))
    return;

  const VarDecl *FirstDecl = cast<VarDecl>(*D->decl_begin());
  // Ensure that there is at least one VarDecl within de DeclStmt.
  assert(FirstDecl && "No VarDecl provided");

  const QualType FirstDeclType = FirstDecl->getType().getCanonicalType();

  std::vector<SourceLocation> StarLocations;
  for (clang::DeclStmt::const_decl_iterator DI = D->decl_begin(),
                                            DE = D->decl_end();
       DI != DE; ++DI) {

    const VarDecl *V = cast<VarDecl>(*DI);
    // Ensure that every DeclStmt child is a VarDecl.
    assert(V && "No VarDecl provided");

    const CXXNewExpr *NewExpr =
        cast<CXXNewExpr>(V->getInit()->IgnoreParenImpCasts());
    // Ensure that every VarDecl has a CXXNewExpr initializer.
    assert(NewExpr && "No CXXNewExpr provided");

    // If VarDecl and Initializer have mismatching unqualified types.
    if (!Result.Context->hasSameUnqualifiedType(V->getType(),
                                                NewExpr->getType()))
      return;

    // Remove explicitly written '*' from declarations where there's more than
    // one declaration in the declaration list.
    if (DI == D->decl_begin())
      continue;

    // All subsequent delcarations should match the same non-decorated type.
    if (FirstDeclType != V->getType().getCanonicalType())
      return;

    PointerTypeLoc Q =
        V->getTypeSourceInfo()->getTypeLoc().getAs<PointerTypeLoc>();
    while (!Q.isNull()) {
      StarLocations.push_back(Q.getStarLoc());
      Q = Q.getNextTypeLoc().getAs<PointerTypeLoc>();
    }
  }

  // Remove '*' from declarations using the saved star locations.
  for (std::vector<SourceLocation>::iterator I = StarLocations.begin(),
                                             E = StarLocations.end();
       I != E; ++I) {
    Replace.insert(tooling::Replacement(SM, *I, 1, ""));
  }
  // FIXME: There is, however, one case we can address: when the VarDecl
  // pointee is the same as the initializer, just more CV-qualified. However,
  // TypeLoc information is not reliable where CV qualifiers are concerned so
  // we can't do anything about this case for now.
  CharSourceRange Range(
      FirstDecl->getTypeSourceInfo()->getTypeLoc().getSourceRange(), true);
  // Space after 'auto' to handle cases where the '*' in the pointer type
  // is next to the identifier. This avoids changing 'int *p' into 'autop'.
  Replace.insert(tooling::Replacement(SM, Range, "auto "));
  ++AcceptedChanges;
}
