//===--- CommentSema.cpp - Doxygen comment semantic analysis --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/CommentSema.h"
#include "llvm/ADT/StringSwitch.h"

namespace clang {
namespace comments {

Sema::Sema(llvm::BumpPtrAllocator &Allocator) :
    Allocator(Allocator) {
}

ParagraphComment *Sema::actOnParagraphComment(
                              ArrayRef<InlineContentComment *> Content) {
  return new (Allocator) ParagraphComment(Content);
}

BlockCommandComment *Sema::actOnBlockCommandStart(SourceLocation LocBegin,
                                                  SourceLocation LocEnd,
                                                  StringRef Name) {
  return new (Allocator) BlockCommandComment(LocBegin, LocEnd, Name);
}

BlockCommandComment *Sema::actOnBlockCommandArgs(
                              BlockCommandComment *Command,
                              ArrayRef<BlockCommandComment::Argument> Args) {
  Command->setArgs(Args);
  return Command;
}

BlockCommandComment *Sema::actOnBlockCommandFinish(
                              BlockCommandComment *Command,
                              ParagraphComment *Paragraph) {
  Command->setParagraph(Paragraph);
  return Command;
}

ParamCommandComment *Sema::actOnParamCommandStart(SourceLocation LocBegin,
                                                  SourceLocation LocEnd,
                                                  StringRef Name) {
  return new (Allocator) ParamCommandComment(LocBegin, LocEnd, Name);
}

ParamCommandComment *Sema::actOnParamCommandArg(ParamCommandComment *Command,
                                                SourceLocation ArgLocBegin,
                                                SourceLocation ArgLocEnd,
                                                StringRef Arg,
                                                bool IsDirection) {
  if (IsDirection) {
    ParamCommandComment::PassDirection Direction;
    std::string ArgLower = Arg.lower();
    // TODO: optimize: lower Name first (need an API in SmallString for that),
    // after that StringSwitch.
    if (ArgLower == "[in]")
      Direction = ParamCommandComment::In;
    else if (ArgLower == "[out]")
      Direction = ParamCommandComment::Out;
    else if (ArgLower == "[in,out]" || ArgLower == "[out,in]")
      Direction = ParamCommandComment::InOut;
    else {
      // Remove spaces.
      std::string::iterator O = ArgLower.begin();
      for (std::string::iterator I = ArgLower.begin(), E = ArgLower.end();
           I != E; ++I) {
        const char C = *I;
        if (C != ' ' && C != '\n' && C != '\r' &&
            C != '\t' && C != '\v' && C != '\f')
          *O++ = C;
      }
      ArgLower.resize(O - ArgLower.begin());

      bool RemovingWhitespaceHelped = false;
      if (ArgLower == "[in]") {
        Direction = ParamCommandComment::In;
        RemovingWhitespaceHelped = true;
      } else if (ArgLower == "[out]") {
        Direction = ParamCommandComment::Out;
        RemovingWhitespaceHelped = true;
      } else if (ArgLower == "[in,out]" || ArgLower == "[out,in]") {
        Direction = ParamCommandComment::InOut;
        RemovingWhitespaceHelped = true;
      } else {
        Direction = ParamCommandComment::In;
        RemovingWhitespaceHelped = false;
      }
      // Diag() unrecognized parameter passing direction, valid directions are ...
      // if (RemovingWhitespaceHelped) FixIt
    }
    Command->setDirection(Direction, /* Explicit = */ true);
  } else {
    if (Command->getArgCount() == 0) {
      if (!Command->isDirectionExplicit()) {
        // User didn't provide a direction argument.
        Command->setDirection(ParamCommandComment::In, /* Explicit = */ false);
      }
      typedef BlockCommandComment::Argument Argument;
      Argument *A = new (Allocator) Argument(SourceRange(ArgLocBegin,
                                                         ArgLocEnd),
                                             Arg);
      Command->setArgs(llvm::makeArrayRef(A, 1));
      // if (...) Diag() unrecognized parameter name
    } else {
      // Diag() \\param command requires at most 2 arguments
    }
  }
  return Command;
}

ParamCommandComment *Sema::actOnParamCommandFinish(ParamCommandComment *Command,
                                                   ParagraphComment *Paragraph) {
  Command->setParagraph(Paragraph);
  return Command;
}

InlineCommandComment *Sema::actOnInlineCommand(SourceLocation CommandLocBegin,
                                               SourceLocation CommandLocEnd,
                                               StringRef CommandName) {
  ArrayRef<InlineCommandComment::Argument> Args;
  return new (Allocator) InlineCommandComment(CommandLocBegin,
                                              CommandLocEnd,
                                              CommandName,
                                              Args);
}

InlineCommandComment *Sema::actOnInlineCommand(SourceLocation CommandLocBegin,
                                               SourceLocation CommandLocEnd,
                                               StringRef CommandName,
                                               SourceLocation ArgLocBegin,
                                               SourceLocation ArgLocEnd,
                                               StringRef Arg) {
  typedef InlineCommandComment::Argument Argument;
  Argument *A = new (Allocator) Argument(SourceRange(ArgLocBegin,
                                                     ArgLocEnd),
                                         Arg);

  return new (Allocator) InlineCommandComment(CommandLocBegin,
                                              CommandLocEnd,
                                              CommandName,
                                              llvm::makeArrayRef(A, 1));
}

InlineContentComment *Sema::actOnUnknownCommand(SourceLocation LocBegin,
                                                SourceLocation LocEnd,
                                                StringRef Name) {
  ArrayRef<InlineCommandComment::Argument> Args;
  return new (Allocator) InlineCommandComment(LocBegin, LocEnd, Name, Args);
}

TextComment *Sema::actOnText(SourceLocation LocBegin,
                             SourceLocation LocEnd,
                             StringRef Text) {
  return new (Allocator) TextComment(LocBegin, LocEnd, Text);
}

VerbatimBlockComment *Sema::actOnVerbatimBlockStart(SourceLocation Loc,
                                                    StringRef Name) {
  return new (Allocator) VerbatimBlockComment(
                                  Loc,
                                  Loc.getLocWithOffset(1 + Name.size()),
                                  Name);
}

VerbatimBlockLineComment *Sema::actOnVerbatimBlockLine(SourceLocation Loc,
                                                       StringRef Text) {
  return new (Allocator) VerbatimBlockLineComment(Loc, Text);
}

VerbatimBlockComment *Sema::actOnVerbatimBlockFinish(
                            VerbatimBlockComment *Block,
                            SourceLocation CloseNameLocBegin,
                            StringRef CloseName,
                            ArrayRef<VerbatimBlockLineComment *> Lines) {
  Block->setCloseName(CloseName, CloseNameLocBegin);
  Block->setLines(Lines);
  return Block;
}

VerbatimLineComment *Sema::actOnVerbatimLine(SourceLocation LocBegin,
                                             StringRef Name,
                                             SourceLocation TextBegin,
                                             StringRef Text) {
  return new (Allocator) VerbatimLineComment(
                              LocBegin,
                              TextBegin.getLocWithOffset(Text.size()),
                              Name,
                              TextBegin,
                              Text);
}

HTMLOpenTagComment *Sema::actOnHTMLOpenTagStart(SourceLocation LocBegin,
                                                StringRef TagName) {
  return new (Allocator) HTMLOpenTagComment(LocBegin, TagName);
}

HTMLOpenTagComment *Sema::actOnHTMLOpenTagFinish(
                              HTMLOpenTagComment *Tag,
                              ArrayRef<HTMLOpenTagComment::Attribute> Attrs,
                              SourceLocation GreaterLoc) {
  Tag->setAttrs(Attrs);
  Tag->setGreaterLoc(GreaterLoc);
  return Tag;
}

HTMLCloseTagComment *Sema::actOnHTMLCloseTag(SourceLocation LocBegin,
                                             SourceLocation LocEnd,
                                             StringRef TagName) {
  return new (Allocator) HTMLCloseTagComment(LocBegin, LocEnd, TagName);
}

FullComment *Sema::actOnFullComment(
                              ArrayRef<BlockContentComment *> Blocks) {
  return new (Allocator) FullComment(Blocks);
}

// TODO: tablegen
bool Sema::isBlockCommand(StringRef Name) {
  return llvm::StringSwitch<bool>(Name)
      .Case("brief", true)
      .Case("result", true)
      .Case("return", true)
      .Case("returns", true)
      .Case("author", true)
      .Case("authors", true)
      .Case("pre", true)
      .Case("post", true)
      .Default(false) || isParamCommand(Name);
}

bool Sema::isParamCommand(StringRef Name) {
  return llvm::StringSwitch<bool>(Name)
      .Case("param", true)
      .Case("arg", true)
      .Default(false);
}

unsigned Sema::getBlockCommandNumArgs(StringRef Name) {
  return llvm::StringSwitch<unsigned>(Name)
      .Case("brief", 0)
      .Case("pre", 0)
      .Case("post", 0)
      .Case("author", 0)
      .Case("authors", 0)
      .Default(0);
}

bool Sema::isInlineCommand(StringRef Name) {
  return llvm::StringSwitch<bool>(Name)
      .Case("c", true)
      .Case("em", true)
      .Default(false);
}

bool Sema::HTMLOpenTagNeedsClosing(StringRef Name) {
  return llvm::StringSwitch<bool>(Name)
      .Case("br", true)
      .Default(true);
}

} // end namespace comments
} // end namespace clang

