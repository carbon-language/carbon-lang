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
#include "clang/Basic/OperatorPrecedence.h"
#include "clang/Format/Format.h"
#include <string>

namespace clang {
class Lexer;
class SourceManager;

namespace format {

enum TokenType {
  TT_BinaryOperator,
  TT_BlockComment,
  TT_CastRParen,
  TT_ConditionalExpr,
  TT_CtorInitializerColon,
  TT_ImplicitStringLiteral,
  TT_LineComment,
  TT_ObjCBlockLParen,
  TT_ObjCDecl,
  TT_ObjCMethodSpecifier,
  TT_ObjCMethodExpr,
  TT_ObjCProperty,
  TT_ObjCSelectorName,
  TT_OverloadedOperator,
  TT_PointerOrReference,
  TT_PureVirtualSpecifier,
  TT_RangeBasedForLoopColon,
  TT_StartOfName,
  TT_TemplateCloser,
  TT_TemplateOpener,
  TT_TrailingUnaryOperator,
  TT_UnaryOperator,
  TT_Unknown
};

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

class AnnotatedToken {
public:
  explicit AnnotatedToken(const FormatToken &FormatTok)
      : FormatTok(FormatTok), Type(TT_Unknown), SpaceRequiredBefore(false),
        CanBreakBefore(false), MustBreakBefore(false),
        ClosesTemplateDeclaration(false), MatchingParen(NULL),
        ParameterCount(1), BindingStrength(0), SplitPenalty(0),
        LongestObjCSelectorName(0), Parent(NULL) {
  }

  bool is(tok::TokenKind Kind) const { return FormatTok.Tok.is(Kind); }
  bool isNot(tok::TokenKind Kind) const { return FormatTok.Tok.isNot(Kind); }

  bool isObjCAtKeyword(tok::ObjCKeywordKind Kind) const {
    return FormatTok.Tok.isObjCAtKeyword(Kind);
  }

  FormatToken FormatTok;

  TokenType Type;

  bool SpaceRequiredBefore;
  bool CanBreakBefore;
  bool MustBreakBefore;

  bool ClosesTemplateDeclaration;

  AnnotatedToken *MatchingParen;

  /// \brief Number of parameters, if this is "(", "[" or "<".
  ///
  /// This is initialized to 1 as we don't need to distinguish functions with
  /// 0 parameters from functions with 1 parameter. Thus, we can simply count
  /// the number of commas.
  unsigned ParameterCount;

  /// \brief The total length of the line up to and including this token.
  unsigned TotalLength;

  // FIXME: Come up with a 'cleaner' concept.
  /// \brief The binding strength of a token. This is a combined value of
  /// operator precedence, parenthesis nesting, etc.
  unsigned BindingStrength;

  /// \brief Penalty for inserting a line break before this token.
  unsigned SplitPenalty;

  /// \brief If this is the first ObjC selector name in an ObjC method
  /// definition or call, this contains the length of the longest name.
  unsigned LongestObjCSelectorName;

  std::vector<AnnotatedToken> Children;
  AnnotatedToken *Parent;

  const AnnotatedToken *getPreviousNoneComment() const {
    AnnotatedToken *Tok = Parent;
    while (Tok != NULL && Tok->is(tok::comment))
      Tok = Tok->Parent;
    return Tok;
  }
};

class AnnotatedLine {
public:
  AnnotatedLine(const UnwrappedLine &Line)
      : First(Line.Tokens.front()), Level(Line.Level),
        InPPDirective(Line.InPPDirective),
        MustBeDeclaration(Line.MustBeDeclaration) {
    assert(!Line.Tokens.empty());
    AnnotatedToken *Current = &First;
    for (std::list<FormatToken>::const_iterator I = ++Line.Tokens.begin(),
                                                E = Line.Tokens.end();
         I != E; ++I) {
      Current->Children.push_back(AnnotatedToken(*I));
      Current->Children[0].Parent = Current;
      Current = &Current->Children[0];
    }
    Last = Current;
  }
  AnnotatedLine(const AnnotatedLine &Other)
      : First(Other.First), Type(Other.Type), Level(Other.Level),
        InPPDirective(Other.InPPDirective),
        MustBeDeclaration(Other.MustBeDeclaration) {
    Last = &First;
    while (!Last->Children.empty()) {
      Last->Children[0].Parent = Last;
      Last = &Last->Children[0];
    }
  }

  AnnotatedToken First;
  AnnotatedToken *Last;

  LineType Type;
  unsigned Level;
  bool InPPDirective;
  bool MustBeDeclaration;
};

inline prec::Level getPrecedence(const AnnotatedToken &Tok) {
  return getBinOpPrecedence(Tok.FormatTok.Tok.getKind(), true, true);
}

/// \brief Determines extra information about the tokens comprising an
/// \c UnwrappedLine.
class TokenAnnotator {
public:
  TokenAnnotator(const FormatStyle &Style, SourceManager &SourceMgr, Lexer &Lex)
      : Style(Style), SourceMgr(SourceMgr), Lex(Lex) {
  }

  void annotate(AnnotatedLine &Line);
  void calculateFormattingInformation(AnnotatedLine &Line);

private:
  /// \brief Calculate the penalty for splitting before \c Tok.
  unsigned splitPenalty(const AnnotatedLine &Line, const AnnotatedToken &Tok);

  bool spaceRequiredBetween(const AnnotatedLine &Line,
                            const AnnotatedToken &Left,
                            const AnnotatedToken &Right);

  bool spaceRequiredBefore(const AnnotatedLine &Line,
                           const AnnotatedToken &Tok);

  bool canBreakBefore(const AnnotatedLine &Line, const AnnotatedToken &Right);

  const FormatStyle &Style;
  SourceManager &SourceMgr;
  Lexer &Lex;
};

} // end namespace format
} // end namespace clang

#endif // LLVM_CLANG_FORMAT_TOKEN_ANNOTATOR_H
