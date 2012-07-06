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

#include "clang/Basic/SourceLocation.h"
#include "clang/AST/Comment.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"

namespace clang {
namespace comments {

class Sema {
  llvm::BumpPtrAllocator &Allocator;

public:
  Sema(llvm::BumpPtrAllocator &Allocator);

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

  ParamCommandComment *actOnParamCommandArg(ParamCommandComment *Command,
                                            SourceLocation ArgLocBegin,
                                            SourceLocation ArgLocEnd,
                                            StringRef Arg,
                                            bool IsDirection);

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

  HTMLOpenTagComment *actOnHTMLOpenTagStart(SourceLocation LocBegin,
                                            StringRef TagName);

  HTMLOpenTagComment *actOnHTMLOpenTagFinish(
                              HTMLOpenTagComment *Tag,
                              ArrayRef<HTMLOpenTagComment::Attribute> Attrs,
                              SourceLocation GreaterLoc);

  HTMLCloseTagComment *actOnHTMLCloseTag(SourceLocation LocBegin,
                                         SourceLocation LocEnd,
                                         StringRef TagName);

  FullComment *actOnFullComment(ArrayRef<BlockContentComment *> Blocks);

  bool isBlockCommand(StringRef Name);
  bool isParamCommand(StringRef Name);
  unsigned getBlockCommandNumArgs(StringRef Name);

  bool isInlineCommand(StringRef Name);
  bool HTMLOpenTagNeedsClosing(StringRef Name);
};

} // end namespace comments
} // end namespace clang

#endif

