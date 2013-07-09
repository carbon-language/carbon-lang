//===--- FormatToken.h - Format C++ code ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the declaration of the FormatToken, a wrapper
/// around Token with additional information related to formatting.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FORMAT_FORMAT_TOKEN_H
#define LLVM_CLANG_FORMAT_FORMAT_TOKEN_H

#include "clang/Basic/OperatorPrecedence.h"
#include "clang/Lex/Lexer.h"

namespace clang {
namespace format {

enum TokenType {
  TT_BinaryOperator,
  TT_BlockComment,
  TT_CastRParen,
  TT_ConditionalExpr,
  TT_CtorInitializerColon,
  TT_DesignatedInitializerPeriod,
  TT_ImplicitStringLiteral,
  TT_InlineASMColon,
  TT_InheritanceColon,
  TT_FunctionTypeLParen,
  TT_LineComment,
  TT_ObjCArrayLiteral,
  TT_ObjCBlockLParen,
  TT_ObjCDecl,
  TT_ObjCDictLiteral,
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

// Represents what type of block a set of braces open.
enum BraceBlockKind {
  BK_Unknown,
  BK_Block,
  BK_BracedInit
};

/// \brief A wrapper around a \c Token storing information about the
/// whitespace characters preceeding it.
struct FormatToken {
  FormatToken()
      : NewlinesBefore(0), HasUnescapedNewline(false), LastNewlineOffset(0),
        CodePointCount(0), IsFirst(false), MustBreakBefore(false),
        BlockKind(BK_Unknown), Type(TT_Unknown), SpacesRequiredBefore(0),
        CanBreakBefore(false), ClosesTemplateDeclaration(false),
        ParameterCount(0), TotalLength(0), UnbreakableTailLength(0),
        BindingStrength(0), SplitPenalty(0), LongestObjCSelectorName(0),
        FakeRParens(0), LastInChainOfCalls(false),
        PartOfMultiVariableDeclStmt(false), MatchingParen(NULL), Previous(NULL),
        Next(NULL) {}

  /// \brief The \c Token.
  Token Tok;

  /// \brief The number of newlines immediately before the \c Token.
  ///
  /// This can be used to determine what the user wrote in the original code
  /// and thereby e.g. leave an empty line between two function definitions.
  unsigned NewlinesBefore;

  /// \brief Whether there is at least one unescaped newline before the \c
  /// Token.
  bool HasUnescapedNewline;

  /// \brief The range of the whitespace immediately preceeding the \c Token.
  SourceRange WhitespaceRange;

  /// \brief The offset just past the last '\n' in this token's leading
  /// whitespace (relative to \c WhiteSpaceStart). 0 if there is no '\n'.
  unsigned LastNewlineOffset;

  /// \brief The length of the non-whitespace parts of the token in CodePoints.
  /// We need this to correctly measure number of columns a token spans.
  unsigned CodePointCount;

  /// \brief Indicates that this is the first token.
  bool IsFirst;

  /// \brief Whether there must be a line break before this token.
  ///
  /// This happens for example when a preprocessor directive ended directly
  /// before the token.
  bool MustBreakBefore;

  /// \brief Returns actual token start location without leading escaped
  /// newlines and whitespace.
  ///
  /// This can be different to Tok.getLocation(), which includes leading escaped
  /// newlines.
  SourceLocation getStartOfNonWhitespace() const {
    return WhitespaceRange.getEnd();
  }

  /// \brief The raw text of the token.
  ///
  /// Contains the raw token text without leading whitespace and without leading
  /// escaped newlines.
  StringRef TokenText;

  /// \brief Contains the kind of block if this token is a brace.
  BraceBlockKind BlockKind;

  TokenType Type;

  unsigned SpacesRequiredBefore;
  bool CanBreakBefore;

  bool ClosesTemplateDeclaration;

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

  /// \brief Stores the number of required fake parentheses and the
  /// corresponding operator precedence.
  ///
  /// If multiple fake parentheses start at a token, this vector stores them in
  /// reverse order, i.e. inner fake parenthesis first.
  SmallVector<prec::Level, 4> FakeLParens;
  /// \brief Insert this many fake ) after this token for correct indentation.
  unsigned FakeRParens;

  /// \brief Is this the last "." or "->" in a builder-type call?
  bool LastInChainOfCalls;

  /// \brief Is this token part of a \c DeclStmt defining multiple variables?
  ///
  /// Only set if \c Type == \c TT_StartOfName.
  bool PartOfMultiVariableDeclStmt;

  bool is(tok::TokenKind Kind) const { return Tok.is(Kind); }

  bool isOneOf(tok::TokenKind K1, tok::TokenKind K2) const {
    return is(K1) || is(K2);
  }

  bool isOneOf(tok::TokenKind K1, tok::TokenKind K2, tok::TokenKind K3) const {
    return is(K1) || is(K2) || is(K3);
  }

  bool isOneOf(tok::TokenKind K1, tok::TokenKind K2, tok::TokenKind K3,
               tok::TokenKind K4, tok::TokenKind K5 = tok::NUM_TOKENS,
               tok::TokenKind K6 = tok::NUM_TOKENS,
               tok::TokenKind K7 = tok::NUM_TOKENS,
               tok::TokenKind K8 = tok::NUM_TOKENS,
               tok::TokenKind K9 = tok::NUM_TOKENS,
               tok::TokenKind K10 = tok::NUM_TOKENS,
               tok::TokenKind K11 = tok::NUM_TOKENS,
               tok::TokenKind K12 = tok::NUM_TOKENS) const {
    return is(K1) || is(K2) || is(K3) || is(K4) || is(K5) || is(K6) || is(K7) ||
           is(K8) || is(K9) || is(K10) || is(K11) || is(K12);
  }

  bool isNot(tok::TokenKind Kind) const { return Tok.isNot(Kind); }

  bool isObjCAtKeyword(tok::ObjCKeywordKind Kind) const {
    return Tok.isObjCAtKeyword(Kind);
  }

  bool isAccessSpecifier(bool ColonRequired = true) const {
    return isOneOf(tok::kw_public, tok::kw_protected, tok::kw_private) &&
           (!ColonRequired || (Next && Next->is(tok::colon)));
  }

  bool isObjCAccessSpecifier() const {
    return is(tok::at) && Next && (Next->isObjCAtKeyword(tok::objc_public) ||
                                   Next->isObjCAtKeyword(tok::objc_protected) ||
                                   Next->isObjCAtKeyword(tok::objc_package) ||
                                   Next->isObjCAtKeyword(tok::objc_private));
  }

  /// \brief Returns whether \p Tok is ([{ or a template opening <.
  bool opensScope() const {
    return isOneOf(tok::l_paren, tok::l_brace, tok::l_square) ||
           Type == TT_TemplateOpener;
  }
  /// \brief Returns whether \p Tok is )]} or a template closing >.
  bool closesScope() const {
    return isOneOf(tok::r_paren, tok::r_brace, tok::r_square) ||
           Type == TT_TemplateCloser;
  }

  bool isUnaryOperator() const {
    switch (Tok.getKind()) {
    case tok::plus:
    case tok::plusplus:
    case tok::minus:
    case tok::minusminus:
    case tok::exclaim:
    case tok::tilde:
    case tok::kw_sizeof:
    case tok::kw_alignof:
      return true;
    default:
      return false;
    }
  }
  bool isBinaryOperator() const {
    // Comma is a binary operator, but does not behave as such wrt. formatting.
    return getPrecedence() > prec::Comma;
  }
  bool isTrailingComment() const {
    return is(tok::comment) && (!Next || Next->NewlinesBefore > 0);
  }

  prec::Level getPrecedence() const {
    return getBinOpPrecedence(Tok.getKind(), true, true);
  }

  /// \brief Returns the previous token ignoring comments.
  FormatToken *getPreviousNonComment() const {
    FormatToken *Tok = Previous;
    while (Tok != NULL && Tok->is(tok::comment))
      Tok = Tok->Previous;
    return Tok;
  }

  /// \brief Returns the next token ignoring comments.
  const FormatToken *getNextNonComment() const {
    const FormatToken *Tok = Next;
    while (Tok != NULL && Tok->is(tok::comment))
      Tok = Tok->Next;
    return Tok;
  }

  FormatToken *MatchingParen;

  FormatToken *Previous;
  FormatToken *Next;

private:
  // Disallow copying.
  FormatToken(const FormatToken &) LLVM_DELETED_FUNCTION;
  void operator=(const FormatToken &) LLVM_DELETED_FUNCTION;
};

} // namespace format
} // namespace clang

#endif // LLVM_CLANG_FORMAT_FORMAT_TOKEN_H
