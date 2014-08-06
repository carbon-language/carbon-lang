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
#include "clang/Format/Format.h"
#include "clang/Lex/Lexer.h"
#include <memory>

namespace clang {
namespace format {

enum TokenType {
  TT_ArrayInitializerLSquare,
  TT_ArraySubscriptLSquare,
  TT_AttributeParen,
  TT_BinaryOperator,
  TT_BitFieldColon,
  TT_BlockComment,
  TT_CastRParen,
  TT_ConditionalExpr,
  TT_ConflictAlternative,
  TT_ConflictEnd,
  TT_ConflictStart,
  TT_CtorInitializerColon,
  TT_CtorInitializerComma,
  TT_DesignatedInitializerPeriod,
  TT_DictLiteral,
  TT_FunctionDeclarationName,
  TT_FunctionLBrace,
  TT_FunctionTypeLParen,
  TT_ImplicitStringLiteral,
  TT_InheritanceColon,
  TT_InlineASMColon,
  TT_LambdaLSquare,
  TT_LineComment,
  TT_ObjCBlockLBrace,
  TT_ObjCBlockLParen,
  TT_ObjCDecl,
  TT_ObjCForIn,
  TT_ObjCMethodExpr,
  TT_ObjCMethodSpecifier,
  TT_ObjCProperty,
  TT_OverloadedOperator,
  TT_OverloadedOperatorLParen,
  TT_PointerOrReference,
  TT_PureVirtualSpecifier,
  TT_RangeBasedForLoopColon,
  TT_RegexLiteral,
  TT_SelectorName,
  TT_StartOfName,
  TT_TemplateCloser,
  TT_TemplateOpener,
  TT_TrailingAnnotation,
  TT_TrailingReturnArrow,
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

// The packing kind of a function's parameters.
enum ParameterPackingKind {
  PPK_BinPacked,
  PPK_OnePerLine,
  PPK_Inconclusive
};

enum FormatDecision {
  FD_Unformatted,
  FD_Continue,
  FD_Break
};

class TokenRole;
class AnnotatedLine;

/// \brief A wrapper around a \c Token storing information about the
/// whitespace characters preceding it.
struct FormatToken {
  FormatToken()
      : NewlinesBefore(0), HasUnescapedNewline(false), LastNewlineOffset(0),
        ColumnWidth(0), LastLineColumnWidth(0), IsMultiline(false),
        IsFirst(false), MustBreakBefore(false), IsUnterminatedLiteral(false),
        BlockKind(BK_Unknown), Type(TT_Unknown), SpacesRequiredBefore(0),
        CanBreakBefore(false), ClosesTemplateDeclaration(false),
        ParameterCount(0), BlockParameterCount(0),
        PackingKind(PPK_Inconclusive), TotalLength(0), UnbreakableTailLength(0),
        BindingStrength(0), NestingLevel(0), SplitPenalty(0),
        LongestObjCSelectorName(0), FakeRParens(0),
        StartsBinaryExpression(false), EndsBinaryExpression(false),
        OperatorIndex(0), LastOperator(false),
        PartOfMultiVariableDeclStmt(false), IsForEachMacro(false),
        MatchingParen(nullptr), Previous(nullptr), Next(nullptr),
        Decision(FD_Unformatted), Finalized(false) {}

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

  /// \brief The range of the whitespace immediately preceding the \c Token.
  SourceRange WhitespaceRange;

  /// \brief The offset just past the last '\n' in this token's leading
  /// whitespace (relative to \c WhiteSpaceStart). 0 if there is no '\n'.
  unsigned LastNewlineOffset;

  /// \brief The width of the non-whitespace parts of the token (or its first
  /// line for multi-line tokens) in columns.
  /// We need this to correctly measure number of columns a token spans.
  unsigned ColumnWidth;

  /// \brief Contains the width in columns of the last line of a multi-line
  /// token.
  unsigned LastLineColumnWidth;

  /// \brief Whether the token text contains newlines (escaped or not).
  bool IsMultiline;

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

  /// \brief Set to \c true if this token is an unterminated literal.
  bool IsUnterminatedLiteral;

  /// \brief Contains the kind of block if this token is a brace.
  BraceBlockKind BlockKind;

  TokenType Type;

  /// \brief The number of spaces that should be inserted before this token.
  unsigned SpacesRequiredBefore;

  /// \brief \c true if it is allowed to break before this token.
  bool CanBreakBefore;

  bool ClosesTemplateDeclaration;

  /// \brief Number of parameters, if this is "(", "[" or "<".
  ///
  /// This is initialized to 1 as we don't need to distinguish functions with
  /// 0 parameters from functions with 1 parameter. Thus, we can simply count
  /// the number of commas.
  unsigned ParameterCount;

  /// \brief Number of parameters that are nested blocks,
  /// if this is "(", "[" or "<".
  unsigned BlockParameterCount;

  /// \brief A token can have a special role that can carry extra information
  /// about the token's formatting.
  std::unique_ptr<TokenRole> Role;

  /// \brief If this is an opening parenthesis, how are the parameters packed?
  ParameterPackingKind PackingKind;

  /// \brief The total length of the unwrapped line up to and including this
  /// token.
  unsigned TotalLength;

  /// \brief The original 0-based column of this token, including expanded tabs.
  /// The configured TabWidth is used as tab width.
  unsigned OriginalColumn;

  /// \brief The length of following tokens until the next natural split point,
  /// or the next token that can be broken.
  unsigned UnbreakableTailLength;

  // FIXME: Come up with a 'cleaner' concept.
  /// \brief The binding strength of a token. This is a combined value of
  /// operator precedence, parenthesis nesting, etc.
  unsigned BindingStrength;

  /// \brief The nesting level of this token, i.e. the number of surrounding (),
  /// [], {} or <>.
  unsigned NestingLevel;

  /// \brief Penalty for inserting a line break before this token.
  unsigned SplitPenalty;

  /// \brief If this is the first ObjC selector name in an ObjC method
  /// definition or call, this contains the length of the longest name.
  ///
  /// This being set to 0 means that the selectors should not be colon-aligned,
  /// e.g. because several of them are block-type.
  unsigned LongestObjCSelectorName;

  /// \brief Stores the number of required fake parentheses and the
  /// corresponding operator precedence.
  ///
  /// If multiple fake parentheses start at a token, this vector stores them in
  /// reverse order, i.e. inner fake parenthesis first.
  SmallVector<prec::Level, 4> FakeLParens;
  /// \brief Insert this many fake ) after this token for correct indentation.
  unsigned FakeRParens;

  /// \brief \c true if this token starts a binary expression, i.e. has at least
  /// one fake l_paren with a precedence greater than prec::Unknown.
  bool StartsBinaryExpression;
  /// \brief \c true if this token ends a binary expression.
  bool EndsBinaryExpression;

  /// \brief Is this is an operator (or "."/"->") in a sequence of operators
  /// with the same precedence, contains the 0-based operator index.
  unsigned OperatorIndex;

  /// \brief Is this the last operator (or "."/"->") in a sequence of operators
  /// with the same precedence?
  bool LastOperator;

  /// \brief Is this token part of a \c DeclStmt defining multiple variables?
  ///
  /// Only set if \c Type == \c TT_StartOfName.
  bool PartOfMultiVariableDeclStmt;

  /// \brief Is this a foreach macro?
  bool IsForEachMacro;

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
  bool isStringLiteral() const { return tok::isStringLiteral(Tok.getKind()); }

  bool isObjCAtKeyword(tok::ObjCKeywordKind Kind) const {
    return Tok.isObjCAtKeyword(Kind);
  }

  bool isAccessSpecifier(bool ColonRequired = true) const {
    return isOneOf(tok::kw_public, tok::kw_protected, tok::kw_private) &&
           (!ColonRequired || (Next && Next->is(tok::colon)));
  }

  /// \brief Determine whether the token is a simple-type-specifier.
  bool isSimpleTypeSpecifier() const;

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

  /// \brief Returns \c true if this is a "." or "->" accessing a member.
  bool isMemberAccess() const {
    return isOneOf(tok::arrow, tok::period, tok::arrowstar) &&
           Type != TT_DesignatedInitializerPeriod;
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

  /// \brief Returns \c true if this is a keyword that can be used
  /// like a function call (e.g. sizeof, typeid, ...).
  bool isFunctionLikeKeyword() const {
    switch (Tok.getKind()) {
    case tok::kw_throw:
    case tok::kw_typeid:
    case tok::kw_return:
    case tok::kw_sizeof:
    case tok::kw_alignof:
    case tok::kw_alignas:
    case tok::kw_decltype:
    case tok::kw_noexcept:
    case tok::kw_static_assert:
    case tok::kw___attribute:
      return true;
    default:
      return false;
    }
  }

  prec::Level getPrecedence() const {
    return getBinOpPrecedence(Tok.getKind(), true, true);
  }

  /// \brief Returns the previous token ignoring comments.
  FormatToken *getPreviousNonComment() const {
    FormatToken *Tok = Previous;
    while (Tok && Tok->is(tok::comment))
      Tok = Tok->Previous;
    return Tok;
  }

  /// \brief Returns the next token ignoring comments.
  const FormatToken *getNextNonComment() const {
    const FormatToken *Tok = Next;
    while (Tok && Tok->is(tok::comment))
      Tok = Tok->Next;
    return Tok;
  }

  /// \brief Returns \c true if this tokens starts a block-type list, i.e. a
  /// list that should be indented with a block indent.
  bool opensBlockTypeList(const FormatStyle &Style) const {
    return Type == TT_ArrayInitializerLSquare ||
           (is(tok::l_brace) &&
            (BlockKind == BK_Block || Type == TT_DictLiteral ||
             !Style.Cpp11BracedListStyle));
  }

  /// \brief Same as opensBlockTypeList, but for the closing token.
  bool closesBlockTypeList(const FormatStyle &Style) const {
    return MatchingParen && MatchingParen->opensBlockTypeList(Style);
  }

  FormatToken *MatchingParen;

  FormatToken *Previous;
  FormatToken *Next;

  SmallVector<AnnotatedLine *, 1> Children;

  /// \brief Stores the formatting decision for the token once it was made.
  FormatDecision Decision;

  /// \brief If \c true, this token has been fully formatted (indented and
  /// potentially re-formatted inside), and we do not allow further formatting
  /// changes.
  bool Finalized;

private:
  // Disallow copying.
  FormatToken(const FormatToken &) LLVM_DELETED_FUNCTION;
  void operator=(const FormatToken &) LLVM_DELETED_FUNCTION;
};

class ContinuationIndenter;
struct LineState;

class TokenRole {
public:
  TokenRole(const FormatStyle &Style) : Style(Style) {}
  virtual ~TokenRole();

  /// \brief After the \c TokenAnnotator has finished annotating all the tokens,
  /// this function precomputes required information for formatting.
  virtual void precomputeFormattingInfos(const FormatToken *Token);

  /// \brief Apply the special formatting that the given role demands.
  ///
  /// Assumes that the token having this role is already formatted.
  ///
  /// Continues formatting from \p State leaving indentation to \p Indenter and
  /// returns the total penalty that this formatting incurs.
  virtual unsigned formatFromToken(LineState &State,
                                   ContinuationIndenter *Indenter,
                                   bool DryRun) {
    return 0;
  }

  /// \brief Same as \c formatFromToken, but assumes that the first token has
  /// already been set thereby deciding on the first line break.
  virtual unsigned formatAfterToken(LineState &State,
                                    ContinuationIndenter *Indenter,
                                    bool DryRun) {
    return 0;
  }

  /// \brief Notifies the \c Role that a comma was found.
  virtual void CommaFound(const FormatToken *Token) {}

protected:
  const FormatStyle &Style;
};

class CommaSeparatedList : public TokenRole {
public:
  CommaSeparatedList(const FormatStyle &Style)
      : TokenRole(Style), HasNestedBracedList(false) {}

  void precomputeFormattingInfos(const FormatToken *Token) override;

  unsigned formatAfterToken(LineState &State, ContinuationIndenter *Indenter,
                            bool DryRun) override;

  unsigned formatFromToken(LineState &State, ContinuationIndenter *Indenter,
                           bool DryRun) override;

  /// \brief Adds \p Token as the next comma to the \c CommaSeparated list.
  void CommaFound(const FormatToken *Token) override {
    Commas.push_back(Token);
  }

private:
  /// \brief A struct that holds information on how to format a given list with
  /// a specific number of columns.
  struct ColumnFormat {
    /// \brief The number of columns to use.
    unsigned Columns;

    /// \brief The total width in characters.
    unsigned TotalWidth;

    /// \brief The number of lines required for this format.
    unsigned LineCount;

    /// \brief The size of each column in characters.
    SmallVector<unsigned, 8> ColumnSizes;
  };

  /// \brief Calculate which \c ColumnFormat fits best into
  /// \p RemainingCharacters.
  const ColumnFormat *getColumnFormat(unsigned RemainingCharacters) const;

  /// \brief The ordered \c FormatTokens making up the commas of this list.
  SmallVector<const FormatToken *, 8> Commas;

  /// \brief The length of each of the list's items in characters including the
  /// trailing comma.
  SmallVector<unsigned, 8> ItemLengths;

  /// \brief Precomputed formats that can be used for this list.
  SmallVector<ColumnFormat, 4> Formats;

  bool HasNestedBracedList;
};

} // namespace format
} // namespace clang

#endif // LLVM_CLANG_FORMAT_FORMAT_TOKEN_H
