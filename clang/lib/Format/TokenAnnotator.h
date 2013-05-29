//===--- TokenAnnotator.h - Format C++ code ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a token annotator, i.e. creates
/// \c AnnotatedTokens out of \c FormatTokens with required extra information.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FORMAT_TOKEN_ANNOTATOR_H
#define LLVM_CLANG_FORMAT_TOKEN_ANNOTATOR_H

#include "UnwrappedLineParser.h"
#include "clang/Format/Format.h"
#include <string>

namespace clang {
class Lexer;
class SourceManager;

namespace format {

enum LineType {
  LT_Invalid,
  LT_Other,
  LT_BuilderTypeCall,
  LT_PreprocessorDirective,
  LT_VirtualFunctionDecl,
  LT_ObjCDecl, // An @interface, @implementation, or @protocol line.
  LT_ObjCMethodDecl,
  LT_ObjCProperty // An @property line.
};

class AnnotatedLine {
public:
  AnnotatedLine(const UnwrappedLine &Line)
      : First(Line.Tokens.front()), Level(Line.Level),
        InPPDirective(Line.InPPDirective),
        MustBeDeclaration(Line.MustBeDeclaration), MightBeFunctionDecl(false),
        StartsDefinition(false) {
    assert(!Line.Tokens.empty());
    FormatToken *Current = First;
    for (std::list<FormatToken *>::const_iterator I = ++Line.Tokens.begin(),
                                                  E = Line.Tokens.end();
         I != E; ++I) {
      Current->Next = *I;
      (*I)->Previous = Current;
      Current = Current->Next;
    }
    Last = Current;
  }

  FormatToken *First;
  FormatToken *Last;

  LineType Type;
  unsigned Level;
  bool InPPDirective;
  bool MustBeDeclaration;
  bool MightBeFunctionDecl;
  bool StartsDefinition;
};

/// \brief Determines extra information about the tokens comprising an
/// \c UnwrappedLine.
class TokenAnnotator {
public:
  TokenAnnotator(const FormatStyle &Style, SourceManager &SourceMgr, Lexer &Lex,
                 IdentifierInfo &Ident_in)
      : Style(Style), SourceMgr(SourceMgr), Lex(Lex), Ident_in(Ident_in) {
  }

  void annotate(AnnotatedLine &Line);
  void calculateFormattingInformation(AnnotatedLine &Line);

private:
  /// \brief Calculate the penalty for splitting before \c Tok.
  unsigned splitPenalty(const AnnotatedLine &Line, const FormatToken &Tok);

  bool spaceRequiredBetween(const AnnotatedLine &Line, const FormatToken &Left,
                            const FormatToken &Right);

  bool spaceRequiredBefore(const AnnotatedLine &Line, const FormatToken &Tok);

  bool canBreakBefore(const AnnotatedLine &Line, const FormatToken &Right);

  void printDebugInfo(const AnnotatedLine &Line);

  void calculateUnbreakableTailLengths(AnnotatedLine &Line);

  const FormatStyle &Style;
  SourceManager &SourceMgr;
  Lexer &Lex;

  // Contextual keywords:
  IdentifierInfo &Ident_in;
};

} // end namespace format
} // end namespace clang

#endif // LLVM_CLANG_FORMAT_TOKEN_ANNOTATOR_H
