//===-- ReplaceAutoPtrActions.cpp --- std::auto_ptr replacement -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the definition of the ASTMatcher callback
/// for the ReplaceAutoPtr transform.
///
//===----------------------------------------------------------------------===//

#include "ReplaceAutoPtrActions.h"
#include "ReplaceAutoPtrMatchers.h"
#include "Core/Transform.h"

#include "clang/AST/ASTContext.h"
#include "clang/Lex/Lexer.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

namespace {

/// \brief Verifies that the token at \p BeginningOfToken is 'auto_ptr'.
bool checkTokenIsAutoPtr(clang::SourceLocation BeginningOfToken,
                         const clang::SourceManager &SM,
                         const clang::LangOptions &LangOptions) {
  llvm::SmallVector<char, 8> Buffer;
  bool Invalid = false;
  llvm::StringRef Res =
      Lexer::getSpelling(BeginningOfToken, Buffer, SM, LangOptions, &Invalid);

  if (Invalid)
    return false;

  return Res == "auto_ptr";
}

} // end anonymous namespace

void AutoPtrReplacer::run(const MatchFinder::MatchResult &Result) {
  SourceManager &SM = *Result.SourceManager;
  SourceLocation IdentifierLoc;

  if (const TypeLoc *TL = Result.Nodes.getNodeAs<TypeLoc>(AutoPtrTokenId)) {
    IdentifierLoc = locateFromTypeLoc(*TL, SM);
  } else {
    const UsingDecl *D = Result.Nodes.getNodeAs<UsingDecl>(AutoPtrTokenId);
    assert(D && "Bad Callback. No node provided.");
    IdentifierLoc = locateFromUsingDecl(D, SM);
  }

  if (IdentifierLoc.isMacroID())
    IdentifierLoc = SM.getSpellingLoc(IdentifierLoc);

  if (!Owner.isFileModifiable(SM, IdentifierLoc))
    return;

  // make sure that only the 'auto_ptr' token is replaced and not the template
  // aliases [temp.alias]
  if (!checkTokenIsAutoPtr(IdentifierLoc, SM, LangOptions()))
    return;

  Replace.insert(
      Replacement(SM, IdentifierLoc, strlen("auto_ptr"), "unique_ptr"));
  ++AcceptedChanges;
}

SourceLocation AutoPtrReplacer::locateFromTypeLoc(TypeLoc AutoPtrTypeLoc,
                                                  const SourceManager &SM) {
  TemplateSpecializationTypeLoc TL =
      AutoPtrTypeLoc.getAs<TemplateSpecializationTypeLoc>();
  if (TL.isNull())
    return SourceLocation();

  return TL.getTemplateNameLoc();
}

SourceLocation
AutoPtrReplacer::locateFromUsingDecl(const UsingDecl *UsingAutoPtrDecl,
                                     const SourceManager &SM) {
  return UsingAutoPtrDecl->getNameInfo().getBeginLoc();
}

void OwnershipTransferFixer::run(const MatchFinder::MatchResult &Result) {
  SourceManager &SM = *Result.SourceManager;
  const Expr *E = Result.Nodes.getNodeAs<Expr>(AutoPtrOwnershipTransferId);
  assert(E && "Bad Callback. No node provided.");

  CharSourceRange Range = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(E->getSourceRange()), SM, LangOptions());

  if (Range.isInvalid())
    return;

  if (!Owner.isFileModifiable(SM, Range.getBegin()))
    return;

  Replace.insert(Replacement(SM, Range.getBegin(), 0, "std::move("));
  Replace.insert(Replacement(SM, Range.getEnd(), 0, ")"));
  AcceptedChanges += 2;
}
