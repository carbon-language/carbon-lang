//===-- nullptr-convert/NullptrActions.cpp - Matcher callback -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  \brief This file contains the definition of the NullptrFixer class which is
///  used as an ASTMatcher callback. Also within this file is a helper AST
///  visitor class used to identify sequences of explicit casts.
///
//===----------------------------------------------------------------------===//

#include "NullptrActions.h"
#include "NullptrMatchers.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace clang;

namespace {

const char *NullMacroName = "NULL";

static llvm::cl::opt<std::string> UserNullMacroNames(
    "user-null-macros", llvm::cl::desc("Comma-separated list of user-defined "
                                       "macro names that behave like NULL"),
    llvm::cl::init(""));

/// \brief Replaces the provided range with the text "nullptr", but only if
/// the start and end location are both in main file.
/// Returns true if and only if a replacement was made.
bool ReplaceWithNullptr(tooling::Replacements &Replace, SourceManager &SM,
                        SourceLocation StartLoc, SourceLocation EndLoc) {
  if (SM.isFromSameFile(StartLoc, EndLoc) && SM.isFromMainFile(StartLoc)) {
    CharSourceRange Range(SourceRange(StartLoc, EndLoc), true);
    Replace.insert(tooling::Replacement(SM, Range, "nullptr"));
    return true;
  } else
    return false;
}

/// \brief Returns the name of the outermost macro.
///
/// Given
/// \code
/// #define MY_NULL NULL
/// \endcode
/// If \p Loc points to NULL, this function will return the name MY_NULL.
llvm::StringRef GetOutermostMacroName(
    SourceLocation Loc, const SourceManager &SM, const LangOptions &LO) {
  assert(Loc.isMacroID());
  SourceLocation OutermostMacroLoc;

  while (Loc.isMacroID()) {
    OutermostMacroLoc = Loc;
    Loc = SM.getImmediateMacroCallerLoc(Loc);
  }

  return clang::Lexer::getImmediateMacroName(OutermostMacroLoc, SM, LO);
}
}

/// \brief Looks for implicit casts as well as sequences of 0 or more explicit
/// casts with an implicit null-to-pointer cast within.
///
/// The matcher this visitor is used with will find a single implicit cast or a
/// top-most explicit cast (i.e. it has no explicit casts as an ancestor) where
/// an implicit cast is nested within. However, there is no guarantee that only
/// explicit casts exist between the found top-most explicit cast and the
/// possibly more than one nested implicit cast. This visitor finds all cast
/// sequences with an implicit cast to null within and creates a replacement
/// leaving the outermost explicit cast unchanged to avoid introducing
/// ambiguities.
class CastSequenceVisitor : public RecursiveASTVisitor<CastSequenceVisitor> {
public:
  CastSequenceVisitor(tooling::Replacements &R, SourceManager &SM,
                      const LangOptions &LangOpts,
                      const UserMacroNames &UserNullMacros,
                      unsigned &AcceptedChanges)
      : Replace(R), SM(SM), LangOpts(LangOpts), UserNullMacros(UserNullMacros),
        AcceptedChanges(AcceptedChanges), FirstSubExpr(0) {}

  // Only VisitStmt is overridden as we shouldn't find other base AST types
  // within a cast expression.
  bool VisitStmt(Stmt *S) {
    CastExpr *C = dyn_cast<CastExpr>(S);

    if (!C) {
      ResetFirstSubExpr();
      return true;
    } else if (!FirstSubExpr) {
      // Keep parentheses for implicit casts to avoid cases where an implicit
      // cast within a parentheses expression is right next to a return
      // statement otherwise get the subexpression of the outermost explicit
      // cast.
      if (C->getStmtClass() == Stmt::ImplicitCastExprClass)
        FirstSubExpr = C->IgnoreParenImpCasts();
      else
        FirstSubExpr = C->getSubExpr();
    }

    if (C->getCastKind() == CK_NullToPointer ||
        C->getCastKind() == CK_NullToMemberPointer) {

      SourceLocation StartLoc = FirstSubExpr->getLocStart();
      SourceLocation EndLoc = FirstSubExpr->getLocEnd();

      // If the start/end location is a macro argument expansion, get the
      // expansion location. If its a macro body expansion, check to see if its
      // coming from a macro called NULL.
      if (SM.isMacroArgExpansion(StartLoc) && SM.isMacroArgExpansion(EndLoc)) {
        StartLoc = SM.getFileLoc(StartLoc);
        EndLoc = SM.getFileLoc(EndLoc);
      } else if (SM.isMacroBodyExpansion(StartLoc) &&
                 SM.isMacroBodyExpansion(EndLoc)) {
        llvm::StringRef OutermostMacroName =
            GetOutermostMacroName(StartLoc, SM, LangOpts);

        // Check to see if the user wants to replace the macro being expanded.
        bool ReplaceNullMacro =
            std::find(UserNullMacros.begin(), UserNullMacros.end(),
                      OutermostMacroName) != UserNullMacros.end();

        if (!ReplaceNullMacro)
          return false;

        StartLoc = SM.getFileLoc(StartLoc);
        EndLoc = SM.getFileLoc(EndLoc);
      }

      AcceptedChanges +=
          ReplaceWithNullptr(Replace, SM, StartLoc, EndLoc) ? 1 : 0;

      ResetFirstSubExpr();
    }

    return true;
  }

private:
  void ResetFirstSubExpr() { FirstSubExpr = 0; }

private:
  tooling::Replacements &Replace;
  SourceManager &SM;
  const LangOptions &LangOpts;
  const UserMacroNames &UserNullMacros;
  unsigned &AcceptedChanges;
  Expr *FirstSubExpr;
};

NullptrFixer::NullptrFixer(clang::tooling::Replacements &Replace,
                           unsigned &AcceptedChanges, RiskLevel)
    : Replace(Replace), AcceptedChanges(AcceptedChanges) {
  if (!UserNullMacroNames.empty()) {
    llvm::StringRef S = UserNullMacroNames;
    S.split(UserNullMacros, ",");
  }
  UserNullMacros.insert(UserNullMacros.begin(), llvm::StringRef(NullMacroName));
}

void NullptrFixer::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  SourceManager &SM = *Result.SourceManager;

  const CastExpr *NullCast = Result.Nodes.getNodeAs<CastExpr>(CastSequence);
  assert(NullCast && "Bad Callback. No node provided");
  // Given an implicit null-ptr cast or an explicit cast with an implicit
  // null-to-pointer cast within use CastSequenceVisitor to identify sequences
  // of explicit casts that can be converted into 'nullptr'.
  CastSequenceVisitor Visitor(Replace, SM, Result.Context->getLangOpts(),
     UserNullMacros, AcceptedChanges);
  Visitor.TraverseStmt(const_cast<CastExpr *>(NullCast));
}
