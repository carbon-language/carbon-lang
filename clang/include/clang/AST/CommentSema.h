//===--- CommentSema.h - Doxygen comment semantic analysis ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the semantic analysis class for Doxygen comments.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_COMMENT_SEMA_H
#define LLVM_CLANG_AST_COMMENT_SEMA_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/AST/Comment.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"

namespace clang {
class Decl;
class FunctionDecl;
class ParmVarDecl;
class SourceMgr;

namespace comments {

class Sema {
  /// Allocator for AST nodes.
  llvm::BumpPtrAllocator &Allocator;

  /// Source manager for the comment being parsed.
  const SourceManager &SourceMgr;

  DiagnosticsEngine &Diags;

  const Decl *ThisDecl;

  DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID) {
    return Diags.Report(Loc, DiagID);
  }

  /// A stack of HTML tags that are currently open (not matched with closing
  /// tags).
  SmallVector<HTMLStartTagComment *, 8> HTMLOpenTags;

public:
  Sema(llvm::BumpPtrAllocator &Allocator, const SourceManager &SourceMgr,
       DiagnosticsEngine &Diags);

  void setDecl(const Decl *D);

  ParagraphComment *actOnParagraphComment(
      ArrayRef<InlineContentComment *> Content);

  BlockCommandComment *actOnBlockCommandStart(SourceLocation LocBegin,
                                              SourceLocation LocEnd,
                                              StringRef Name);

  BlockCommandComment *actOnBlockCommandArgs(
                              BlockCommandComment *Command,
                              ArrayRef<BlockCommandComment::Argument> Args);

  BlockCommandComment *actOnBlockCommandFinish(BlockCommandComment *Command,
                                               ParagraphComment *Paragraph);

  ParamCommandComment *actOnParamCommandStart(SourceLocation LocBegin,
                                              SourceLocation LocEnd,
                                              StringRef Name);

  ParamCommandComment *actOnParamCommandDirectionArg(
                                            ParamCommandComment *Command,
                                            SourceLocation ArgLocBegin,
                                            SourceLocation ArgLocEnd,
                                            StringRef Arg);

  ParamCommandComment *actOnParamCommandParamNameArg(
                                            ParamCommandComment *Command,
                                            SourceLocation ArgLocBegin,
                                            SourceLocation ArgLocEnd,
                                            StringRef Arg);

  ParamCommandComment *actOnParamCommandFinish(ParamCommandComment *Command,
                                               ParagraphComment *Paragraph);

  InlineCommandComment *actOnInlineCommand(SourceLocation CommandLocBegin,
                                           SourceLocation CommandLocEnd,
                                           StringRef CommandName);

  InlineCommandComment *actOnInlineCommand(SourceLocation CommandLocBegin,
                                           SourceLocation CommandLocEnd,
                                           StringRef CommandName,
                                           SourceLocation ArgLocBegin,
                                           SourceLocation ArgLocEnd,
                                           StringRef Arg);

  InlineContentComment *actOnUnknownCommand(SourceLocation LocBegin,
                                            SourceLocation LocEnd,
                                            StringRef Name);

  TextComment *actOnText(SourceLocation LocBegin,
                         SourceLocation LocEnd,
                         StringRef Text);

  VerbatimBlockComment *actOnVerbatimBlockStart(SourceLocation Loc,
                                                StringRef Name);

  VerbatimBlockLineComment *actOnVerbatimBlockLine(SourceLocation Loc,
                                                   StringRef Text);

  VerbatimBlockComment *actOnVerbatimBlockFinish(
                              VerbatimBlockComment *Block,
                              SourceLocation CloseNameLocBegin,
                              StringRef CloseName,
                              ArrayRef<VerbatimBlockLineComment *> Lines);

  VerbatimLineComment *actOnVerbatimLine(SourceLocation LocBegin,
                                         StringRef Name,
                                         SourceLocation TextBegin,
                                         StringRef Text);

  HTMLStartTagComment *actOnHTMLStartTagStart(SourceLocation LocBegin,
                                              StringRef TagName);

  HTMLStartTagComment *actOnHTMLStartTagFinish(
                              HTMLStartTagComment *Tag,
                              ArrayRef<HTMLStartTagComment::Attribute> Attrs,
                              SourceLocation GreaterLoc,
                              bool IsSelfClosing);

  HTMLEndTagComment *actOnHTMLEndTag(SourceLocation LocBegin,
                                     SourceLocation LocEnd,
                                     StringRef TagName);

  FullComment *actOnFullComment(ArrayRef<BlockContentComment *> Blocks);

  void checkBlockCommandEmptyParagraph(BlockCommandComment *Command);

  /// Returns index of a function parameter with a given name.
  unsigned resolveParmVarReference(StringRef Name,
                                   const ParmVarDecl * const *ParamVars,
                                   unsigned NumParams);

  /// Returns index of a function parameter with the name closest to a given
  /// typo.
  unsigned correctTypoInParmVarReference(StringRef Typo,
                                         const ParmVarDecl * const *ParamVars,
                                         unsigned NumParams);

  bool isBlockCommand(StringRef Name);
  bool isParamCommand(StringRef Name);
  unsigned getBlockCommandNumArgs(StringRef Name);

  bool isInlineCommand(StringRef Name);

  bool isHTMLEndTagOptional(StringRef Name);
  bool isHTMLEndTagForbidden(StringRef Name);
};

} // end namespace comments
} // end namespace clang

#endif

