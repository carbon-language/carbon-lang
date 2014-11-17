//===--- NamespaceCommentCheck.cpp - clang-tidy ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NamespaceCommentCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

NamespaceCommentCheck::NamespaceCommentCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      NamespaceCommentPattern("^/[/*] *(end (of )?)? *(anonymous|unnamed)? *"
                              "namespace( +([a-zA-Z0-9_]+))?\\.? *(\\*/)?$",
                              llvm::Regex::IgnoreCase),
      ShortNamespaceLines(Options.get("ShortNamespaceLines", 1u)),
      SpacesBeforeComments(Options.get("SpacesBeforeComments", 1u)) {}

void NamespaceCommentCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ShortNamespaceLines", ShortNamespaceLines);
  Options.store(Opts, "SpacesBeforeComments", SpacesBeforeComments);
}

void NamespaceCommentCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(namespaceDecl().bind("namespace"), this);
}

bool locationsInSameFile(const SourceManager &Sources, SourceLocation Loc1,
                         SourceLocation Loc2) {
  return Loc1.isFileID() && Loc2.isFileID() &&
         Sources.getFileID(Loc1) == Sources.getFileID(Loc2);
}

std::string getNamespaceComment(const NamespaceDecl *ND, bool InsertLineBreak) {
  std::string Fix = "// namespace";
  if (!ND->isAnonymousNamespace())
    Fix.append(" ").append(ND->getNameAsString());
  if (InsertLineBreak)
    Fix.append("\n");
  return Fix;
}

void NamespaceCommentCheck::check(const MatchFinder::MatchResult &Result) {
  const NamespaceDecl *ND = Result.Nodes.getNodeAs<NamespaceDecl>("namespace");
  const SourceManager &Sources = *Result.SourceManager;

  if (!locationsInSameFile(Sources, ND->getLocStart(), ND->getRBraceLoc()))
    return;

  // Don't require closing comments for namespaces spanning less than certain
  // number of lines.
  unsigned StartLine = Sources.getSpellingLineNumber(ND->getLocStart());
  unsigned EndLine = Sources.getSpellingLineNumber(ND->getRBraceLoc());
  if (EndLine - StartLine + 1 <= ShortNamespaceLines)
    return;

  // Find next token after the namespace closing brace.
  SourceLocation AfterRBrace = ND->getRBraceLoc().getLocWithOffset(1);
  SourceLocation Loc = AfterRBrace;
  Token Tok;
  // Skip whitespace until we find the next token.
  while (Lexer::getRawToken(Loc, Tok, Sources, Result.Context->getLangOpts())) {
    Loc = Loc.getLocWithOffset(1);
  }
  if (!locationsInSameFile(Sources, ND->getRBraceLoc(), Loc))
    return;

  bool NextTokenIsOnSameLine = Sources.getSpellingLineNumber(Loc) == EndLine;
  // If we insert a line comment before the token in the same line, we need
  // to insert a line break.
  bool NeedLineBreak = NextTokenIsOnSameLine && Tok.isNot(tok::eof);

  // Try to find existing namespace closing comment on the same line.
  if (Tok.is(tok::comment) && NextTokenIsOnSameLine) {
    StringRef Comment(Sources.getCharacterData(Loc), Tok.getLength());
    SmallVector<StringRef, 7> Groups;
    if (NamespaceCommentPattern.match(Comment, &Groups)) {
      StringRef NamespaceNameInComment = Groups.size() > 5 ? Groups[5] : "";
      StringRef Anonymous = Groups.size() > 3 ? Groups[3] : "";

      // Check if the namespace in the comment is the same.
      if ((ND->isAnonymousNamespace() && NamespaceNameInComment.empty()) ||
          (ND->getNameAsString() == NamespaceNameInComment &&
           Anonymous.empty())) {
        // FIXME: Maybe we need a strict mode, where we always fix namespace
        // comments with different format.
        return;
      }

      // Otherwise we need to fix the comment.
      NeedLineBreak = Comment.startswith("/*");
      CharSourceRange OldCommentRange = CharSourceRange::getCharRange(
          SourceRange(Loc, Loc.getLocWithOffset(Tok.getLength())));
      diag(Loc, "namespace closing comment refers to a wrong namespace '%0'")
          << NamespaceNameInComment
          << FixItHint::CreateReplacement(
                 OldCommentRange, getNamespaceComment(ND, NeedLineBreak));
      return;
    }

    // This is not a recognized form of a namespace closing comment.
    // Leave line comment on the same line. Move block comment to the next line,
    // as it can be multi-line or there may be other tokens behind it.
    if (Comment.startswith("//"))
      NeedLineBreak = false;
  }

  std::string NamespaceName =
      ND->isAnonymousNamespace()
          ? "anonymous namespace"
          : ("namespace '" + ND->getNameAsString() + "'");

  diag(AfterRBrace, "%0 not terminated with a closing comment")
      << NamespaceName
      << FixItHint::CreateInsertion(AfterRBrace,
                                    std::string(SpacesBeforeComments, ' ') +
                                        getNamespaceComment(ND, NeedLineBreak));
  diag(ND->getLocation(), "%0 starts here", DiagnosticIDs::Note)
      << NamespaceName;
}

} // namespace readability
} // namespace tidy
} // namespace clang
