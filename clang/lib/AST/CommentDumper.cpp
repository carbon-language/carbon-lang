//===--- CommentDumper.cpp - Dumping implementation for Comment ASTs ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/CommentVisitor.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace comments {

namespace {
class CommentDumper: public comments::ConstCommentVisitor<CommentDumper> {
  raw_ostream &OS;
  SourceManager *SM;
  unsigned IndentLevel;

public:
  CommentDumper(raw_ostream &OS, SourceManager *SM) :
      OS(OS), SM(SM), IndentLevel(0)
  { }

  void dumpIndent() const {
    for (unsigned i = 1, e = IndentLevel; i < e; ++i)
      OS << "  ";
  }

  void dumpLocation(SourceLocation Loc) {
    if (SM)
      Loc.print(OS, *SM);
  }

  void dumpSourceRange(const Comment *C);

  void dumpComment(const Comment *C);

  void dumpSubtree(const Comment *C);

  // Inline content.
  void visitTextComment(const TextComment *C);
  void visitInlineCommandComment(const InlineCommandComment *C);
  void visitHTMLOpenTagComment(const HTMLOpenTagComment *C);
  void visitHTMLCloseTagComment(const HTMLCloseTagComment *C);

  // Block content.
  void visitParagraphComment(const ParagraphComment *C);
  void visitBlockCommandComment(const BlockCommandComment *C);
  void visitParamCommandComment(const ParamCommandComment *C);
  void visitVerbatimBlockComment(const VerbatimBlockComment *C);
  void visitVerbatimBlockLineComment(const VerbatimBlockLineComment *C);
  void visitVerbatimLineComment(const VerbatimLineComment *C);

  void visitFullComment(const FullComment *C);
};

void CommentDumper::dumpSourceRange(const Comment *C) {
  if (!SM)
    return;

  SourceRange SR = C->getSourceRange();

  OS << " <";
  dumpLocation(SR.getBegin());
  if (SR.getBegin() != SR.getEnd()) {
    OS << ", ";
    dumpLocation(SR.getEnd());
  }
  OS << ">";
}

void CommentDumper::dumpComment(const Comment *C) {
  dumpIndent();
  OS << "(" << C->getCommentKindName()
     << " " << (void *) C;
  dumpSourceRange(C);
}

void CommentDumper::dumpSubtree(const Comment *C) {
  ++IndentLevel;
  if (C) {
    visit(C);
    for (Comment::child_iterator I = C->child_begin(),
                                 E = C->child_end();
         I != E; ++I) {
      OS << '\n';
      dumpSubtree(*I);
    }
    OS << ')';
  } else {
    dumpIndent();
    OS << "<<<NULL>>>";
  }
  --IndentLevel;
}

void CommentDumper::visitTextComment(const TextComment *C) {
  dumpComment(C);

  OS << " Text=\"" << C->getText() << "\"";
}

void CommentDumper::visitInlineCommandComment(const InlineCommandComment *C) {
  dumpComment(C);

  for (unsigned i = 0, e = C->getArgCount(); i != e; ++i)
    OS << " Arg[" << i << "]=\"" << C->getArgText(i) << "\"";
}

void CommentDumper::visitHTMLOpenTagComment(const HTMLOpenTagComment *C) {
  dumpComment(C);

  OS << " Name=\"" << C->getTagName() << "\"";
  if (C->getAttrCount() != 0) {
    OS << " Attrs: ";
    for (unsigned i = 0, e = C->getAttrCount(); i != e; ++i) {
      const HTMLOpenTagComment::Attribute &Attr = C->getAttr(i);
      OS << " \"" << Attr.Name << "=\"" << Attr.Value << "\"";
    }
  }
}

void CommentDumper::visitHTMLCloseTagComment(const HTMLCloseTagComment *C) {
  dumpComment(C);

  OS << " Name=\"" << C->getTagName() << "\"";
}

void CommentDumper::visitParagraphComment(const ParagraphComment *C) {
  dumpComment(C);
}

void CommentDumper::visitBlockCommandComment(const BlockCommandComment *C) {
  dumpComment(C);

  OS << " Name=\"" << C->getCommandName() << "\"";
}

void CommentDumper::visitParamCommandComment(const ParamCommandComment *C) {
  dumpComment(C);

  switch (C->getDirection()) {
  case ParamCommandComment::In:
    OS << " [in]";
    break;
  case ParamCommandComment::Out:
    OS << " [out]";
    break;
  case ParamCommandComment::InOut:
    OS << " [in,out]";
    break;
  }

  if (C->isDirectionExplicit())
    OS << " explicitly";
  else
    OS << " implicitly";

  if (C->hasParamName()) {
    OS << " Param=\"" << C->getParamName() << "\"";
  }
}

void CommentDumper::visitVerbatimBlockComment(const VerbatimBlockComment *C) {
  dumpComment(C);

  OS << " Name=\"" << C->getCommandName() << "\""
        " CloseName=\"" << C->getCloseName() << "\"";
}

void CommentDumper::visitVerbatimBlockLineComment(const VerbatimBlockLineComment *C) {
  dumpComment(C);

  OS << " Text=\"" << C->getText() << "\"";
}

void CommentDumper::visitVerbatimLineComment(const VerbatimLineComment *C) {
  dumpComment(C);

  OS << " Text=\"" << C->getText() << "\"";
}

void CommentDumper::visitFullComment(const FullComment *C) {
  dumpComment(C);
}

} // unnamed namespace

void Comment::dump() const {
  CommentDumper D(llvm::errs(), NULL);
  D.dumpSubtree(this);
  llvm::errs() << '\n';
}

void Comment::dump(SourceManager &SM) const {
  CommentDumper D(llvm::errs(), &SM);
  D.dumpSubtree(this);
  llvm::errs() << '\n';
}

} // end namespace comments
} // end namespace clang

