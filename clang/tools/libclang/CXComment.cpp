//===- CXComment.cpp - libclang APIs for manipulating CXComments ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all libclang APIs related to walking comment AST.
//
//===----------------------------------------------------------------------===//

#include "clang-c/Index.h"
#include "CXString.h"
#include "CXComment.h"

#include "clang/AST/CommentVisitor.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::cxstring;
using namespace clang::comments;
using namespace clang::cxcomment;

extern "C" {

enum CXCommentKind clang_Comment_getKind(CXComment CXC) {
  const Comment *C = getASTNode(CXC);
  if (!C)
    return CXComment_Null;

  switch (C->getCommentKind()) {
  case Comment::NoCommentKind:
    return CXComment_Null;

  case Comment::TextCommentKind:
    return CXComment_Text;

  case Comment::InlineCommandCommentKind:
    return CXComment_InlineCommand;

  case Comment::HTMLStartTagCommentKind:
    return CXComment_HTMLStartTag;

  case Comment::HTMLEndTagCommentKind:
    return CXComment_HTMLEndTag;

  case Comment::ParagraphCommentKind:
    return CXComment_Paragraph;

  case Comment::BlockCommandCommentKind:
    return CXComment_BlockCommand;

  case Comment::ParamCommandCommentKind:
    return CXComment_ParamCommand;

  case Comment::VerbatimBlockCommentKind:
    return CXComment_VerbatimBlockCommand;

  case Comment::VerbatimBlockLineCommentKind:
    return CXComment_VerbatimBlockLine;

  case Comment::VerbatimLineCommentKind:
    return CXComment_VerbatimLine;

  case Comment::FullCommentKind:
    return CXComment_FullComment;
  }
  llvm_unreachable("unknown CommentKind");
}

unsigned clang_Comment_getNumChildren(CXComment CXC) {
  const Comment *C = getASTNode(CXC);
  if (!C)
    return 0;

  return C->child_count();
}

CXComment clang_Comment_getChild(CXComment CXC, unsigned ChildIdx) {
  const Comment *C = getASTNode(CXC);
  if (!C || ChildIdx >= C->child_count())
    return createCXComment(NULL);

  return createCXComment(*(C->child_begin() + ChildIdx));
}

unsigned clang_Comment_isWhitespace(CXComment CXC) {
  const Comment *C = getASTNode(CXC);
  if (!C)
    return false;

  if (const TextComment *TC = dyn_cast<TextComment>(C))
    return TC->isWhitespace();

  if (const ParagraphComment *PC = dyn_cast<ParagraphComment>(C))
    return PC->isWhitespace();

  return false;
}

unsigned clang_InlineContentComment_hasTrailingNewline(CXComment CXC) {
  const InlineContentComment *ICC = getASTNodeAs<InlineContentComment>(CXC);
  if (!ICC)
    return false;

  return ICC->hasTrailingNewline();
}

CXString clang_TextComment_getText(CXComment CXC) {
  const TextComment *TC = getASTNodeAs<TextComment>(CXC);
  if (!TC)
    return createCXString((const char *) 0);

  return createCXString(TC->getText(), /*DupString=*/ false);
}

CXString clang_InlineCommandComment_getCommandName(CXComment CXC) {
  const InlineCommandComment *ICC = getASTNodeAs<InlineCommandComment>(CXC);
  if (!ICC)
    return createCXString((const char *) 0);

  return createCXString(ICC->getCommandName(), /*DupString=*/ false);
}

unsigned clang_InlineCommandComment_getNumArgs(CXComment CXC) {
  const InlineCommandComment *ICC = getASTNodeAs<InlineCommandComment>(CXC);
  if (!ICC)
    return 0;

  return ICC->getNumArgs();
}

CXString clang_InlineCommandComment_getArgText(CXComment CXC,
                                               unsigned ArgIdx) {
  const InlineCommandComment *ICC = getASTNodeAs<InlineCommandComment>(CXC);
  if (!ICC || ArgIdx >= ICC->getNumArgs())
    return createCXString((const char *) 0);

  return createCXString(ICC->getArgText(ArgIdx), /*DupString=*/ false);
}

CXString clang_HTMLTagComment_getTagName(CXComment CXC) {
  const HTMLTagComment *HTC = getASTNodeAs<HTMLTagComment>(CXC);
  if (!HTC)
    return createCXString((const char *) 0);

  return createCXString(HTC->getTagName(), /*DupString=*/ false);
}

unsigned clang_HTMLStartTagComment_isSelfClosing(CXComment CXC) {
  const HTMLStartTagComment *HST = getASTNodeAs<HTMLStartTagComment>(CXC);
  if (!HST)
    return false;

  return HST->isSelfClosing();
}

unsigned clang_HTMLStartTag_getNumAttrs(CXComment CXC) {
  const HTMLStartTagComment *HST = getASTNodeAs<HTMLStartTagComment>(CXC);
  if (!HST)
    return 0;

  return HST->getNumAttrs();
}

CXString clang_HTMLStartTag_getAttrName(CXComment CXC, unsigned AttrIdx) {
  const HTMLStartTagComment *HST = getASTNodeAs<HTMLStartTagComment>(CXC);
  if (!HST || AttrIdx >= HST->getNumAttrs())
    return createCXString((const char *) 0);

  return createCXString(HST->getAttr(AttrIdx).Name, /*DupString=*/ false);
}

CXString clang_HTMLStartTag_getAttrValue(CXComment CXC, unsigned AttrIdx) {
  const HTMLStartTagComment *HST = getASTNodeAs<HTMLStartTagComment>(CXC);
  if (!HST || AttrIdx >= HST->getNumAttrs())
    return createCXString((const char *) 0);

  return createCXString(HST->getAttr(AttrIdx).Value, /*DupString=*/ false);
}

CXString clang_BlockCommandComment_getCommandName(CXComment CXC) {
  const BlockCommandComment *BCC = getASTNodeAs<BlockCommandComment>(CXC);
  if (!BCC)
    return createCXString((const char *) 0);

  return createCXString(BCC->getCommandName(), /*DupString=*/ false);
}

unsigned clang_BlockCommandComment_getNumArgs(CXComment CXC) {
  const BlockCommandComment *BCC = getASTNodeAs<BlockCommandComment>(CXC);
  if (!BCC)
    return 0;

  return BCC->getNumArgs();
}

CXString clang_BlockCommandComment_getArgText(CXComment CXC,
                                              unsigned ArgIdx) {
  const BlockCommandComment *BCC = getASTNodeAs<BlockCommandComment>(CXC);
  if (!BCC || ArgIdx >= BCC->getNumArgs())
    return createCXString((const char *) 0);

  return createCXString(BCC->getArgText(ArgIdx), /*DupString=*/ false);
}

CXComment clang_BlockCommandComment_getParagraph(CXComment CXC) {
  const BlockCommandComment *BCC = getASTNodeAs<BlockCommandComment>(CXC);
  if (!BCC)
    return createCXComment(NULL);

  return createCXComment(BCC->getParagraph());
}

CXString clang_ParamCommandComment_getParamName(CXComment CXC) {
  const ParamCommandComment *PCC = getASTNodeAs<ParamCommandComment>(CXC);
  if (!PCC)
    return createCXString((const char *) 0);

  return createCXString(PCC->getParamName(), /*DupString=*/ false);
}

unsigned clang_ParamCommandComment_isParamIndexValid(CXComment CXC) {
  const ParamCommandComment *PCC = getASTNodeAs<ParamCommandComment>(CXC);
  if (!PCC)
    return false;

  return PCC->isParamIndexValid();
}

unsigned clang_ParamCommandComment_getParamIndex(CXComment CXC) {
  const ParamCommandComment *PCC = getASTNodeAs<ParamCommandComment>(CXC);
  if (!PCC)
    return ParamCommandComment::InvalidParamIndex;

  return PCC->getParamIndex();
}

unsigned clang_ParamCommandComment_isDirectionExplicit(CXComment CXC) {
  const ParamCommandComment *PCC = getASTNodeAs<ParamCommandComment>(CXC);
  if (!PCC)
    return false;

  return PCC->isDirectionExplicit();
}

enum CXCommentParamPassDirection clang_ParamCommandComment_getDirection(
                                                            CXComment CXC) {
  const ParamCommandComment *PCC = getASTNodeAs<ParamCommandComment>(CXC);
  if (!PCC)
    return CXCommentParamPassDirection_In;

  switch (PCC->getDirection()) {
  case ParamCommandComment::In:
    return CXCommentParamPassDirection_In;

  case ParamCommandComment::Out:
    return CXCommentParamPassDirection_Out;

  case ParamCommandComment::InOut:
    return CXCommentParamPassDirection_InOut;
  }
  llvm_unreachable("unknown ParamCommandComment::PassDirection");
}

CXString clang_VerbatimBlockLineComment_getText(CXComment CXC) {
  const VerbatimBlockLineComment *VBL =
      getASTNodeAs<VerbatimBlockLineComment>(CXC);
  if (!VBL)
    return createCXString((const char *) 0);

  return createCXString(VBL->getText(), /*DupString=*/ false);
}

CXString clang_VerbatimLineComment_getText(CXComment CXC) {
  const VerbatimLineComment *VLC = getASTNodeAs<VerbatimLineComment>(CXC);
  if (!VLC)
    return createCXString((const char *) 0);

  return createCXString(VLC->getText(), /*DupString=*/ false);
}

} // end extern "C"

//===----------------------------------------------------------------------===//
// Helpers for converting comment AST to HTML.
//===----------------------------------------------------------------------===//

namespace {

class ParamCommandCommentCompareIndex {
public:
  bool operator()(const ParamCommandComment *LHS,
                  const ParamCommandComment *RHS) const {
    // To sort invalid (unresolved) parameters last, this comparison relies on
    // invalid indices to be UINT_MAX.
    return LHS->getParamIndex() < RHS->getParamIndex();
  }
};

class CommentASTToHTMLConverter :
    public ConstCommentVisitor<CommentASTToHTMLConverter> {
public:
  /// \param Str accumulator for HTML.
  CommentASTToHTMLConverter(SmallVectorImpl<char> &Str) : Result(Str) { }

  // Inline content.
  void visitTextComment(const TextComment *C);
  void visitInlineCommandComment(const InlineCommandComment *C);
  void visitHTMLStartTagComment(const HTMLStartTagComment *C);
  void visitHTMLEndTagComment(const HTMLEndTagComment *C);

  // Block content.
  void visitParagraphComment(const ParagraphComment *C);
  void visitBlockCommandComment(const BlockCommandComment *C);
  void visitParamCommandComment(const ParamCommandComment *C);
  void visitVerbatimBlockComment(const VerbatimBlockComment *C);
  void visitVerbatimBlockLineComment(const VerbatimBlockLineComment *C);
  void visitVerbatimLineComment(const VerbatimLineComment *C);

  void visitFullComment(const FullComment *C);

  // Helpers.

  /// Convert a paragraph that is not a block by itself (an argument to some
  /// command).
  void visitNonStandaloneParagraphComment(const ParagraphComment *C);

  void appendToResultWithHTMLEscaping(StringRef S);

private:
  /// Output stream for HTML.
  llvm::raw_svector_ostream Result;
};
} // end unnamed namespace

void CommentASTToHTMLConverter::visitTextComment(const TextComment *C) {
  appendToResultWithHTMLEscaping(C->getText());
}

void CommentASTToHTMLConverter::visitInlineCommandComment(
                                  const InlineCommandComment *C) {
  StringRef CommandName = C->getCommandName();
  bool HasArg0 = C->getNumArgs() > 0 && !C->getArgText(0).empty();
  StringRef Arg0;
  if (HasArg0)
    Arg0 = C->getArgText(0);

  if (CommandName == "b") {
    if (!HasArg0)
      return;
    Result << "<b>" << Arg0 << "</b>";
    return;
  }
  if (CommandName == "c" || CommandName == "p") {
    if (!HasArg0)
      return;
    Result << "<tt>" << Arg0 << "</tt>";
    return;
  }
  if (CommandName == "a" || CommandName == "e" || CommandName == "em") {
    if (!HasArg0)
      return;
    Result << "<em>" << Arg0 << "</em>";
    return;
  }

  // We don't recognize this command, so just print its arguments.
  for (unsigned i = 0, e = C->getNumArgs(); i != e; ++i)
    Result << C->getArgText(i) << " ";
}

void CommentASTToHTMLConverter::visitHTMLStartTagComment(
                                  const HTMLStartTagComment *C) {
  Result << "<" << C->getTagName();

  if (C->getNumAttrs() != 0) {
    for (unsigned i = 0, e = C->getNumAttrs(); i != e; i++) {
      Result << " ";
      const HTMLStartTagComment::Attribute &Attr = C->getAttr(i);
      Result << Attr.Name;
      if (!Attr.Value.empty())
        Result << "=\"" << Attr.Value << "\"";
    }
  }

  if (!C->isSelfClosing())
    Result << ">";
  else
    Result << "/>";
}

void CommentASTToHTMLConverter::visitHTMLEndTagComment(
                                  const HTMLEndTagComment *C) {
  Result << "</" << C->getTagName() << ">";
}

void CommentASTToHTMLConverter::visitParagraphComment(
                                  const ParagraphComment *C) {
  if (C->isWhitespace())
    return;

  Result << "<p>";
  for (Comment::child_iterator I = C->child_begin(), E = C->child_end();
       I != E; ++I) {
    visit(*I);
  }
  Result << "</p>";
}

void CommentASTToHTMLConverter::visitBlockCommandComment(
                                  const BlockCommandComment *C) {
  StringRef CommandName = C->getCommandName();
  if (CommandName == "brief" || CommandName == "short") {
    Result << "<p class=\"para-brief\">";
    visitNonStandaloneParagraphComment(C->getParagraph());
    Result << "</p>";
    return;
  }
  if (CommandName == "returns" || CommandName == "return") {
    Result << "<p class=\"para-returns\">"
              "<span class=\"word-returns\">Returns</span> ";
    visitNonStandaloneParagraphComment(C->getParagraph());
    Result << "</p>";
    return;
  }
  // We don't know anything about this command.  Just render the paragraph.
  visit(C->getParagraph());
}

void CommentASTToHTMLConverter::visitParamCommandComment(
                                  const ParamCommandComment *C) {
  if (C->isParamIndexValid()) {
    Result << "<dt class=\"param-name-index-"
           << C->getParamIndex()
           << "\">";
  } else
    Result << "<dt class=\"param-name-index-invalid\">";

  Result << C->getParamName() << "</dt>";

  if (C->isParamIndexValid()) {
    Result << "<dd class=\"param-descr-index-"
           << C->getParamIndex()
           << "\">";
  } else
    Result << "<dd class=\"param-descr-index-invalid\">";

  visitNonStandaloneParagraphComment(C->getParagraph());
  Result << "</dd>";
}

void CommentASTToHTMLConverter::visitVerbatimBlockComment(
                                  const VerbatimBlockComment *C) {
  unsigned NumLines = C->getNumLines();
  if (NumLines == 0)
    return;

  Result << "<pre>";
  for (unsigned i = 0; i != NumLines; ++i) {
    appendToResultWithHTMLEscaping(C->getText(i));
    if (i + 1 != NumLines)
      Result << '\n';
  }
  Result << "</pre>";
}

void CommentASTToHTMLConverter::visitVerbatimBlockLineComment(
                                  const VerbatimBlockLineComment *C) {
  llvm_unreachable("should not see this AST node");
}

void CommentASTToHTMLConverter::visitVerbatimLineComment(
                                  const VerbatimLineComment *C) {
  Result << "<pre>";
  appendToResultWithHTMLEscaping(C->getText());
  Result << "</pre>";
}

void CommentASTToHTMLConverter::visitFullComment(const FullComment *C) {
  const BlockContentComment *Brief = NULL;
  const ParagraphComment *FirstParagraph = NULL;
  const BlockCommandComment *Returns = NULL;
  SmallVector<const ParamCommandComment *, 8> Params;
  SmallVector<const BlockContentComment *, 8> MiscBlocks;

  // Extract various blocks into separate variables and vectors above.
  for (Comment::child_iterator I = C->child_begin(), E = C->child_end();
       I != E; ++I) {
    const Comment *Child = *I;
    if (!Child)
      continue;
    switch (Child->getCommentKind()) {
    case Comment::NoCommentKind:
      continue;

    case Comment::ParagraphCommentKind: {
      const ParagraphComment *PC = cast<ParagraphComment>(Child);
      if (PC->isWhitespace())
        break;
      if (!FirstParagraph)
        FirstParagraph = PC;

      MiscBlocks.push_back(PC);
      break;
    }

    case Comment::BlockCommandCommentKind: {
      const BlockCommandComment *BCC = cast<BlockCommandComment>(Child);
      StringRef CommandName = BCC->getCommandName();
      if (!Brief && (CommandName == "brief" || CommandName == "short")) {
        Brief = BCC;
        break;
      }
      if (!Returns && (CommandName == "returns" || CommandName == "return")) {
        Returns = BCC;
        break;
      }
      MiscBlocks.push_back(BCC);
      break;
    }

    case Comment::ParamCommandCommentKind: {
      const ParamCommandComment *PCC = cast<ParamCommandComment>(Child);
      if (!PCC->hasParamName())
        break;

      if (!PCC->isDirectionExplicit() && !PCC->hasNonWhitespaceParagraph())
        break;

      Params.push_back(PCC);
      break;
    }

    case Comment::VerbatimBlockCommentKind:
    case Comment::VerbatimLineCommentKind:
      MiscBlocks.push_back(cast<BlockCommandComment>(Child));
      break;

    case Comment::TextCommentKind:
    case Comment::InlineCommandCommentKind:
    case Comment::HTMLStartTagCommentKind:
    case Comment::HTMLEndTagCommentKind:
    case Comment::VerbatimBlockLineCommentKind:
    case Comment::FullCommentKind:
      llvm_unreachable("AST node of this kind can't be a child of "
                       "a FullComment");
    }
  }

  // Sort params in order they are declared in the function prototype.
  // Unresolved parameters are put at the end of the list in the same order
  // they were seen in the comment.
  std::stable_sort(Params.begin(), Params.end(),
                   ParamCommandCommentCompareIndex());

  bool FirstParagraphIsBrief = false;
  if (Brief)
    visit(Brief);
  else if (FirstParagraph) {
    Result << "<p class=\"para-brief\">";
    visitNonStandaloneParagraphComment(FirstParagraph);
    Result << "</p>";
    FirstParagraphIsBrief = true;
  }

  for (unsigned i = 0, e = MiscBlocks.size(); i != e; ++i) {
    const Comment *C = MiscBlocks[i];
    if (FirstParagraphIsBrief && C == FirstParagraph)
      continue;
    visit(C);
  }

  if (Params.size() != 0) {
    Result << "<dl>";
    for (unsigned i = 0, e = Params.size(); i != e; ++i)
      visit(Params[i]);
    Result << "</dl>";
  }

  if (Returns)
    visit(Returns);

  Result.flush();
}

void CommentASTToHTMLConverter::visitNonStandaloneParagraphComment(
                                  const ParagraphComment *C) {
  if (!C)
    return;

  for (Comment::child_iterator I = C->child_begin(), E = C->child_end();
       I != E; ++I) {
    visit(*I);
  }
}

void CommentASTToHTMLConverter::appendToResultWithHTMLEscaping(StringRef S) {
  for (StringRef::iterator I = S.begin(), E = S.end(); I != E; ++I) {
    const char C = *I;
    switch (C) {
      case '&':
        Result << "&amp;";
        break;
      case '<':
        Result << "&lt;";
        break;
      case '>':
        Result << "&gt;";
        break;
      case '"':
        Result << "&quot;";
        break;
      case '\'':
        Result << "&#39;";
        break;
      case '/':
        Result << "&#47;";
        break;
      default:
        Result << C;
        break;
    }
  }
}

extern "C" {

CXString clang_HTMLTagComment_getAsString(CXComment CXC) {
  const HTMLTagComment *HTC = getASTNodeAs<HTMLTagComment>(CXC);
  if (!HTC)
    return createCXString((const char *) 0);

  SmallString<128> HTML;
  CommentASTToHTMLConverter Converter(HTML);
  Converter.visit(HTC);
  return createCXString(HTML.str(), /* DupString = */ true);
}

CXString clang_FullComment_getAsHTML(CXComment CXC) {
  const FullComment *FC = getASTNodeAs<FullComment>(CXC);
  if (!FC)
    return createCXString((const char *) 0);

  SmallString<1024> HTML;
  CommentASTToHTMLConverter Converter(HTML);
  Converter.visit(FC);
  return createCXString(HTML.str(), /* DupString = */ true);
}

} // end extern "C"

