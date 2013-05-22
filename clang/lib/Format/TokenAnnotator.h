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
  TT_InlineASMColon,
  TT_InheritanceColon,
  TT_LineComment,
  TT_ObjCArrayLiteral,
  TT_ObjCBlockLParen,
  TT_ObjCDecl,
  TT_ObjCForIn,
  TT_ObjCMethodExpr,
  TT_ObjCMethodSpecifier,
  TT_ObjCProperty,
  TT_ObjCSelectorName,
  TT_OverloadedOperator,
  TT_OverloadedOperatorLParen,
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
      : FormatTok(FormatTok), Type(TT_Unknown), SpacesRequiredBefore(0),
        CanBreakBefore(false), MustBreakBefore(false),
        ClosesTemplateDeclaration(false), MatchingParen(NULL),
        ParameterCount(0), TotalLength(FormatTok.TokenLength),
        UnbreakableTailLength(0), BindingStrength(0), SplitPenalty(0),
        LongestObjCSelectorName(0), DefinesFunctionType(false), Parent(NULL),
        FakeRParens(0), LastInChainOfCalls(false),
        PartOfMultiVariableDeclStmt(false) {}

  bool is(tok::TokenKind Kind) const { return FormatTok.Tok.is(Kind); }

  bool isOneOf(tok::TokenKind K1, tok::TokenKind K2) const {
    return is(K1) || is(K2);
  }

  bool isOneOf(tok::TokenKind K1, tok::TokenKind K2, tok::TokenKind K3) const {
    return is(K1) || is(K2) || is(K3);
  }

  bool isOneOf(
      tok::TokenKind K1, tok::TokenKind K2, tok::TokenKind K3,
      tok::TokenKind K4, tok::TokenKind K5 = tok::NUM_TOKENS,
      tok::TokenKind K6 = tok::NUM_TOKENS, tok::TokenKind K7 = tok::NUM_TOKENS,
      tok::TokenKind K8 = tok::NUM_TOKENS, tok::TokenKind K9 = tok::NUM_TOKENS,
      tok::TokenKind K10 = tok::NUM_TOKENS,
      tok::TokenKind K11 = tok::NUM_TOKENS,
      tok::TokenKind K12 = tok::NUM_TOKENS) const {
    return is(K1) || is(K2) || is(K3) || is(K4) || is(K5) || is(K6) || is(K7) ||
           is(K8) || is(K9) || is(K10) || is(K11) || is(K12);
  }

  bool isNot(tok::TokenKind Kind) const { return FormatTok.Tok.isNot(Kind); }

  bool isObjCAtKeyword(tok::ObjCKeywordKind Kind) const {
    return FormatTok.Tok.isObjCAtKeyword(Kind);
  }

  bool isAccessSpecifier(bool ColonRequired = true) const {
    return isOneOf(tok::kw_public, tok::kw_protected, tok::kw_private) &&
           (!ColonRequired ||
            (!Children.empty() && Children[0].is(tok::colon)));
  }

  bool isObjCAccessSpecifier() const {
    return is(tok::at) && !Children.empty() &&
           (Children[0].isObjCAtKeyword(tok::objc_public) ||
            Children[0].isObjCAtKeyword(tok::objc_protected) ||
            Children[0].isObjCAtKeyword(tok::objc_package) ||
            Children[0].isObjCAtKeyword(tok::objc_private));
  }

  /// \brief Returns whether \p Tok is ([{ or a template opening <.
  bool opensScope() const;
  /// \brief Returns whether \p Tok is )]} or a template opening >.
  bool closesScope() const;

  bool isUnaryOperator() const;
  bool isBinaryOperator() const;
  bool isTrailingComment() const;

  FormatToken FormatTok;

  TokenType Type;

  unsigned SpacesRequiredBefore;
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

  /// \brief The length of following tokens until the next natural split point,
  /// or the next token that can be broken.
  unsigned UnbreakableTailLength;

  // FIXME: Come up with a 'cleaner' concept.
  /// \brief The binding strength of a token. This is a combined value of
  /// operator precedence, parenthesis nesting, etc.
  unsigned BindingStrength;

  /// \brief Penalty for inserting a line break before this token.
  unsigned SplitPenalty;

  /// \brief If this is the first ObjC selector name in an ObjC method
  /// definition or call, this contains the length of the longest name.
  unsigned LongestObjCSelectorName;

  /// \brief \c true if this is a "(" that starts a function type definition.
  bool DefinesFunctionType;

  std::vector<AnnotatedToken> Children;
  AnnotatedToken *Parent;

  /// \brief Stores the number of required fake parentheses and the
  /// corresponding operator precedence.
  ///
  /// If multiple fake parentheses start at a token, this vector stores them in
  /// reverse order, i.e. inner fake parenthesis first.
  SmallVector<prec::Level, 4>  FakeLParens;
  /// \brief Insert this many fake ) after this token for correct indentation.
  unsigned FakeRParens;

  /// \brief Is this the last "." or "->" in a builder-type call?
  bool LastInChainOfCalls;

  /// \brief Is this token part of a \c DeclStmt defining multiple variables?
  ///
  /// Only set if \c Type == \c TT_StartOfName.
  bool PartOfMultiVariableDeclStmt;

  /// \brief Returns the previous token ignoring comments.
  AnnotatedToken *getPreviousNoneComment() const;

  /// \brief Returns the next token ignoring comments.
  const AnnotatedToken *getNextNoneComment() const;
};

class AnnotatedLine {
public:
  AnnotatedLine(const UnwrappedLine &Line)
      : First(Line.Tokens.front()), Level(Line.Level),
        InPPDirective(Line.InPPDirective),
        MustBeDeclaration(Line.MustBeDeclaration), MightBeFunctionDecl(false),
        StartsDefinition(false) {
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
        MustBeDeclaration(Other.MustBeDeclaration),
        MightBeFunctionDecl(Other.MightBeFunctionDecl),
        StartsDefinition(Other.StartsDefinition) {
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
  bool MightBeFunctionDecl;
  bool StartsDefinition;
};

inline prec::Level getPrecedence(const AnnotatedToken &Tok) {
  return getBinOpPrecedence(Tok.FormatTok.Tok.getKind(), true, true);
}

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
  unsigned splitPenalty(const AnnotatedLine &Line, const AnnotatedToken &Tok);

  bool spaceRequiredBetween(const AnnotatedLine &Line,
                            const AnnotatedToken &Left,
                            const AnnotatedToken &Right);

  bool spaceRequiredBefore(const AnnotatedLine &Line,
                           const AnnotatedToken &Tok);

  bool canBreakBefore(const AnnotatedLine &Line, const AnnotatedToken &Right);

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
