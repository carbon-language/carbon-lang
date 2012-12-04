//===--- Format.cpp - Format C++ code -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements functions declared in Format.h. This will be
/// split into separate files as we go.
///
/// This is EXPERIMENTAL code under heavy development. It is not in a state yet,
/// where it can be used to format real code.
///
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"
#include "UnwrappedLineParser.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

#include <string>

namespace clang {
namespace format {

// FIXME: Move somewhere sane.
struct TokenAnnotation {
  enum TokenType { TT_Unknown, TT_TemplateOpener, TT_TemplateCloser,
      TT_BinaryOperator, TT_UnaryOperator, TT_OverloadedOperator,
      TT_PointerOrReference, TT_ConditionalExpr, TT_LineComment,
      TT_BlockComment };

  TokenType Type;

  /// \brief The current parenthesis level, i.e. the number of opening minus
  /// the number of closing parenthesis left of the current position.
  unsigned ParenLevel;

  bool SpaceRequiredBefore;
  bool CanBreakBefore;
  bool MustBreakBefore;
};

using llvm::MutableArrayRef;

FormatStyle getLLVMStyle() {
  FormatStyle LLVMStyle;
  LLVMStyle.ColumnLimit = 80;
  LLVMStyle.MaxEmptyLinesToKeep = 1;
  LLVMStyle.PointerAndReferenceBindToType = false;
  LLVMStyle.AccessModifierOffset = -2;
  LLVMStyle.SplitTemplateClosingGreater = true;
  return LLVMStyle;
}

FormatStyle getGoogleStyle() {
  FormatStyle GoogleStyle;
  GoogleStyle.ColumnLimit = 80;
  GoogleStyle.MaxEmptyLinesToKeep = 1;
  GoogleStyle.PointerAndReferenceBindToType = true;
  GoogleStyle.AccessModifierOffset = -1;
  GoogleStyle.SplitTemplateClosingGreater = false;
  return GoogleStyle;
}

struct OptimizationParameters {
  unsigned PenaltyExtraLine;
  unsigned PenaltyIndentLevel;
};

class UnwrappedLineFormatter {
public:
  UnwrappedLineFormatter(const FormatStyle &Style, SourceManager &SourceMgr,
                         const UnwrappedLine &Line,
                         const std::vector<TokenAnnotation> &Annotations,
                         tooling::Replacements &Replaces)
      : Style(Style),
        SourceMgr(SourceMgr),
        Line(Line),
        Annotations(Annotations),
        Replaces(Replaces) {
    Parameters.PenaltyExtraLine = 100;
    Parameters.PenaltyIndentLevel = 5;
  }

  void format() {
    formatFirstToken();
    count = 0;
    IndentState State;
    State.Column = Line.Level * 2 + Line.Tokens[0].Tok.getLength();
    State.CtorInitializerOnNewLine = false;
    State.InCtorInitializer = false;
    State.ConsumedTokens = 1;

    //State.UsedIndent.push_back(Line.Level * 2);
    State.Indent.push_back(Line.Level * 2 + 4);
    State.LastSpace.push_back(Line.Level * 2);

    // Start iterating at 1 as we have correctly formatted of Token #0 above.
    for (unsigned i = 1, n = Line.Tokens.size(); i != n; ++i) {
      unsigned NoBreak = calcPenalty(State, false, UINT_MAX);
      unsigned Break = calcPenalty(State, true, NoBreak);
      addToken(Break < NoBreak, false, State);
    }
  }

private:
  /// \brief The current state when indenting a unwrapped line.
  ///
  /// As the indenting tries different combinations this is copied by value.
  struct IndentState {
    /// \brief The number of used columns in the current line.
    unsigned Column;

    /// \brief The number of tokens already consumed.
    unsigned ConsumedTokens;

    /// \brief The position to which a specific parenthesis level needs to be
    /// indented.
    std::vector<unsigned> Indent;

    std::vector<unsigned> LastSpace;

    bool CtorInitializerOnNewLine;
    bool InCtorInitializer;

    /// \brief Comparison operator to be able to used \c IndentState in \c map.
    bool operator<(const IndentState &Other) const {
      if (Other.ConsumedTokens != ConsumedTokens)
        return Other.ConsumedTokens > ConsumedTokens;
      if (Other.Column != Column)
        return Other.Column > Column;
      if (Other.Indent.size() != Indent.size())
        return Other.Indent.size() > Indent.size();
      for (int i = 0, e = Indent.size(); i != e; ++i) {
        if (Other.Indent[i] != Indent[i])
          return Other.Indent[i] > Indent[i];
      }
      if (Other.LastSpace.size() != LastSpace.size())
        return Other.LastSpace.size() > LastSpace.size();
      for (int i = 0, e = LastSpace.size(); i != e; ++i) {
        if (Other.LastSpace[i] != LastSpace[i])
          return Other.LastSpace[i] > LastSpace[i];
      }
      return false;
    }
  };

  /// Append the next token to \p State.
  void addToken(bool Newline, bool DryRun, IndentState &State) {
    unsigned Index = State.ConsumedTokens;
    const FormatToken &Current = Line.Tokens[Index];
    const FormatToken &Previous = Line.Tokens[Index - 1];
    unsigned ParenLevel = Annotations[Index].ParenLevel;

    if (Current.Tok.is(tok::l_paren) || Current.Tok.is(tok::l_square) ||
        Annotations[Index].Type == TokenAnnotation::TT_TemplateOpener) {
      State.Indent.push_back(4 + State.LastSpace.back());
      State.LastSpace.push_back(State.LastSpace.back());
    }

    if (Newline) {
      if (Current.Tok.is(tok::string_literal) &&
          Previous.Tok.is(tok::string_literal))
        State.Column = State.Column - Previous.Tok.getLength();
      else if (Previous.Tok.is(tok::equal) && ParenLevel != 0)
        State.Column = State.Indent[ParenLevel] + 4;
      else
        State.Column = State.Indent[ParenLevel];
      if (!DryRun)
        replaceWhitespace(Current, 1, State.Column);

      State.Column += Current.Tok.getLength();
      State.LastSpace[ParenLevel] = State.Indent[ParenLevel];
      if (Current.Tok.is(tok::colon) &&
          Annotations[Index].Type != TokenAnnotation::TT_ConditionalExpr) {
        State.Indent[ParenLevel] += 2;
        State.CtorInitializerOnNewLine = true;
        State.InCtorInitializer = true;
      }
    } else {
      unsigned Spaces = Annotations[Index].SpaceRequiredBefore ? 1 : 0;
      if (Annotations[Index].Type == TokenAnnotation::TT_LineComment)
        Spaces = 2;
      if (!DryRun)
        replaceWhitespace(Current, 0, Spaces);
      if (Previous.Tok.is(tok::l_paren))
        State.Indent[ParenLevel] = State.Column;
      if (Annotations[Index - 1].Type == TokenAnnotation::TT_TemplateOpener)
        State.Indent[ParenLevel] = State.Column;
      if (Current.Tok.is(tok::colon)) {
        State.Indent[ParenLevel] = State.Column + 3;
        State.InCtorInitializer = true;
      }
      // Top-level spaces are exempt as that mostly leads to better results.
      if (Spaces > 0 && ParenLevel != 0)
        State.LastSpace[ParenLevel] = State.Column + Spaces;
      State.Column += Current.Tok.getLength() + Spaces;
    }

    if (Current.Tok.is(tok::r_paren) || Current.Tok.is(tok::r_square) ||
        Annotations[Index].Type == TokenAnnotation::TT_TemplateCloser) {
      State.Indent.pop_back();
      State.LastSpace.pop_back();
    }

    ++State.ConsumedTokens;
  }

  typedef std::map<IndentState, unsigned> StateMap;
  StateMap Memory;

  unsigned splitPenalty(const FormatToken &Token) {
    if (Token.Tok.is(tok::semi))
      return 0;
    if (Token.Tok.is(tok::comma))
      return 1;
    if (Token.Tok.is(tok::equal) || Token.Tok.is(tok::l_paren) ||
        Token.Tok.is(tok::pipepipe) || Token.Tok.is(tok::ampamp))
      return 2;
    return 3;
  }

  /// \brief Calculate the number of lines needed to format the remaining part
  /// of the unwrapped line.
  ///
  /// Assumes the formatting so far has led to
  /// the \c IndentState \p State. If \p NewLine is set, a new line will be
  /// added after the previous token.
  ///
  /// \param StopAt is used for optimization. If we can determine that we'll
  /// definitely need at least \p StopAt additional lines, we already know of a
  /// better solution.
  unsigned calcPenalty(IndentState State, bool NewLine, unsigned StopAt) {
    // We are at the end of the unwrapped line, so we don't need any more lines.
    if (State.ConsumedTokens >= Line.Tokens.size())
      return 0;

    if (!NewLine && Annotations[State.ConsumedTokens].MustBreakBefore)
      return UINT_MAX;
    if (NewLine && !Annotations[State.ConsumedTokens].CanBreakBefore)
      return UINT_MAX;

    if (State.ConsumedTokens > 0 && !NewLine &&
        State.CtorInitializerOnNewLine &&
        Line.Tokens[State.ConsumedTokens - 1].Tok.is(tok::comma))
      return UINT_MAX;

    if (NewLine && State.InCtorInitializer && !State.CtorInitializerOnNewLine)
      return UINT_MAX;

    addToken(NewLine, true, State);

    // Exceeding column limit is bad.
    if (State.Column > Style.ColumnLimit)
      return UINT_MAX;

    unsigned CurrentPenalty = 0;
    if (NewLine) {
      CurrentPenalty += Parameters.PenaltyIndentLevel *
          Annotations[State.ConsumedTokens - 1].ParenLevel +
          Parameters.PenaltyExtraLine +
          splitPenalty(Line.Tokens[State.ConsumedTokens - 2]);
    }

    if (StopAt <= CurrentPenalty)
      return UINT_MAX;
    StopAt -= CurrentPenalty;

    // Has this state already been examined?
    StateMap::iterator I = Memory.find(State);
    if (I != Memory.end())
      return I->second;
    ++count;

    unsigned NoBreak = calcPenalty(State, false, StopAt);
    unsigned WithBreak = calcPenalty(State, true, std::min(StopAt, NoBreak));
    unsigned Result = std::min(NoBreak, WithBreak);
    if (Result != UINT_MAX)
      Result += CurrentPenalty;
    Memory[State] = Result;
    assert(Memory.find(State) != Memory.end());
    return Result;
  }

  /// \brief Replaces the whitespace in front of \p Tok. Only call once for
  /// each \c FormatToken.
  void replaceWhitespace(const FormatToken &Tok, unsigned NewLines,
                         unsigned Spaces) {
    Replaces.insert(tooling::Replacement(
        SourceMgr, Tok.WhiteSpaceStart, Tok.WhiteSpaceLength,
        std::string(NewLines, '\n') + std::string(Spaces, ' ')));
  }

  /// \brief Add a new line and the required indent before the first Token
  /// of the \c UnwrappedLine.
  void formatFirstToken() {
    const FormatToken &Token = Line.Tokens[0];
    if (Token.WhiteSpaceStart.isValid()) {
      unsigned Newlines =
          std::min(Token.NewlinesBefore, Style.MaxEmptyLinesToKeep + 1);
      unsigned Offset = SourceMgr.getFileOffset(Token.WhiteSpaceStart);
      if (Newlines == 0 && Offset != 0)
        Newlines = 1;
      unsigned Indent = Line.Level * 2;
      if (Token.Tok.is(tok::kw_public) || Token.Tok.is(tok::kw_protected) ||
          Token.Tok.is(tok::kw_private))
        Indent += Style.AccessModifierOffset;
      replaceWhitespace(Token, Newlines, Indent);
    }
  }

  FormatStyle Style;
  SourceManager &SourceMgr;
  const UnwrappedLine &Line;
  const std::vector<TokenAnnotation> &Annotations;
  tooling::Replacements &Replaces;
  unsigned int count;

  OptimizationParameters Parameters;
};

/// \brief Determines extra information about the tokens comprising an
/// \c UnwrappedLine.
class TokenAnnotator {
public:
  TokenAnnotator(const UnwrappedLine &Line, const FormatStyle &Style,
                 SourceManager &SourceMgr)
      : Line(Line),
        Style(Style),
        SourceMgr(SourceMgr) {
  }

  /// \brief A parser that gathers additional information about tokens.
  ///
  /// The \c TokenAnnotator tries to matches parenthesis and square brakets and
  /// store a parenthesis levels. It also tries to resolve matching "<" and ">"
  /// into template parameter lists.
  class AnnotatingParser {
  public:
    AnnotatingParser(const SmallVector<FormatToken, 16> &Tokens,
                     std::vector<TokenAnnotation> &Annotations)
        : Tokens(Tokens),
          Annotations(Annotations),
          Index(0) {
    }

    bool parseAngle(unsigned Level) {
      while (Index < Tokens.size()) {
        if (Tokens[Index].Tok.is(tok::greater)) {
          Annotations[Index].Type = TokenAnnotation::TT_TemplateCloser;
          Annotations[Index].ParenLevel = Level;
          next();
          return true;
        }
        if (Tokens[Index].Tok.is(tok::r_paren) ||
            Tokens[Index].Tok.is(tok::r_square))
          return false;
        if (Tokens[Index].Tok.is(tok::pipepipe) ||
            Tokens[Index].Tok.is(tok::ampamp) ||
            Tokens[Index].Tok.is(tok::question) ||
            Tokens[Index].Tok.is(tok::colon))
          return false;
        consumeToken(Level);
      }
      return false;
    }

    bool parseParens(unsigned Level) {
      while (Index < Tokens.size()) {
        if (Tokens[Index].Tok.is(tok::r_paren)) {
          Annotations[Index].ParenLevel = Level;
          next();
          return true;
        }
        if (Tokens[Index].Tok.is(tok::r_square))
          return false;
        consumeToken(Level);
      }
      return false;
    }

    bool parseSquare(unsigned Level) {
      while (Index < Tokens.size()) {
        if (Tokens[Index].Tok.is(tok::r_square)) {
          Annotations[Index].ParenLevel = Level;
          next();
          return true;
        }
        if (Tokens[Index].Tok.is(tok::r_paren))
          return false;
        consumeToken(Level);
      }
      return false;
    }

    bool parseConditional(unsigned Level) {
      while (Index < Tokens.size()) {
        if (Tokens[Index].Tok.is(tok::colon)) {
          Annotations[Index].Type = TokenAnnotation::TT_ConditionalExpr;
          next();
          return true;
        }
        consumeToken(Level);
      }
      return false;
    }

    void consumeToken(unsigned Level) {
      Annotations[Index].ParenLevel = Level;
      unsigned CurrentIndex = Index;
      next();
      switch (Tokens[CurrentIndex].Tok.getKind()) {
      case tok::l_paren:
        parseParens(Level + 1);
        break;
      case tok::l_square:
        parseSquare(Level + 1);
        break;
      case tok::less:
        if (parseAngle(Level + 1))
          Annotations[CurrentIndex].Type = TokenAnnotation::TT_TemplateOpener;
        else {
          Annotations[CurrentIndex].ParenLevel = Level;
          Annotations[CurrentIndex].Type = TokenAnnotation::TT_BinaryOperator;
          Index = CurrentIndex + 1;
        }
        break;
      case tok::greater:
        Annotations[CurrentIndex].Type = TokenAnnotation::TT_BinaryOperator;
        break;
      case tok::kw_operator:
        if (!Tokens[Index].Tok.is(tok::l_paren))
          Annotations[Index].Type = TokenAnnotation::TT_OverloadedOperator;
        next();
        break;
      case tok::question:
        parseConditional(Level);
        break;
      default:
        break;
      }
    }

    void parseLine() {
      while (Index < Tokens.size()) {
        consumeToken(0);
      }
    }

    void next() {
      ++Index;
    }

  private:
    const SmallVector<FormatToken, 16> &Tokens;
    std::vector<TokenAnnotation> &Annotations;
    unsigned Index;
  };

  void annotate() {
    Annotations.clear();
    for (int i = 0, e = Line.Tokens.size(); i != e; ++i) {
      Annotations.push_back(TokenAnnotation());
    }

    AnnotatingParser Parser(Line.Tokens, Annotations);
    Parser.parseLine();

    determineTokenTypes();

    for (int i = 1, e = Line.Tokens.size(); i != e; ++i) {
      TokenAnnotation &Annotation = Annotations[i];

      Annotation.CanBreakBefore =
          canBreakBetween(Line.Tokens[i - 1], Line.Tokens[i]);

      if (Line.Tokens[i].Tok.is(tok::colon)) {
        if (Line.Tokens[0].Tok.is(tok::kw_case) || i == e - 1) {
          Annotation.SpaceRequiredBefore = false;
        } else {
          Annotation.SpaceRequiredBefore = TokenAnnotation::TT_ConditionalExpr;
        }
      } else if (Annotations[i - 1].Type == TokenAnnotation::TT_UnaryOperator) {
        Annotation.SpaceRequiredBefore = false;
      } else if (Annotation.Type == TokenAnnotation::TT_UnaryOperator) {
        Annotation.SpaceRequiredBefore =
            Line.Tokens[i - 1].Tok.isNot(tok::l_paren) &&
            Line.Tokens[i - 1].Tok.isNot(tok::l_square);
      } else if (Line.Tokens[i - 1].Tok.is(tok::greater) &&
                 Line.Tokens[i].Tok.is(tok::greater)) {
        if (Annotation.Type == TokenAnnotation::TT_TemplateCloser &&
                 Annotations[i - 1].Type == TokenAnnotation::TT_TemplateCloser)
          Annotation.SpaceRequiredBefore = Style.SplitTemplateClosingGreater;
        else
          Annotation.SpaceRequiredBefore = false;
      } else if (
          Annotation.Type == TokenAnnotation::TT_BinaryOperator ||
          Annotations[i - 1].Type == TokenAnnotation::TT_BinaryOperator) {
        Annotation.SpaceRequiredBefore = true;
      } else if (
          Annotations[i - 1].Type == TokenAnnotation::TT_TemplateCloser &&
          Line.Tokens[i].Tok.is(tok::l_paren)) {
        Annotation.SpaceRequiredBefore = false;
      } else if (Line.Tokens[i].Tok.is(tok::less) &&
                 Line.Tokens[0].Tok.is(tok::hash)) {
        Annotation.SpaceRequiredBefore = true;
      } else {
        Annotation.SpaceRequiredBefore =
            spaceRequiredBetween(Line.Tokens[i - 1].Tok, Line.Tokens[i].Tok);
      }

      if (Annotations[i - 1].Type == TokenAnnotation::TT_LineComment ||
          (Line.Tokens[i].Tok.is(tok::string_literal) &&
           Line.Tokens[i - 1].Tok.is(tok::string_literal))) {
        Annotation.MustBreakBefore = true;
      }

      if (Annotation.MustBreakBefore)
        Annotation.CanBreakBefore = true;
    }
  }

  const std::vector<TokenAnnotation> &getAnnotations() {
    return Annotations;
  }

private:
  void determineTokenTypes() {
    for (int i = 0, e = Line.Tokens.size(); i != e; ++i) {
      TokenAnnotation &Annotation = Annotations[i];
      const FormatToken &Tok = Line.Tokens[i];

      if (Tok.Tok.is(tok::star) || Tok.Tok.is(tok::amp))
        Annotation.Type = determineStarAmpUsage(i);
      else if (isUnaryOperator(i))
        Annotation.Type = TokenAnnotation::TT_UnaryOperator;
      else if (isBinaryOperator(Line.Tokens[i]))
        Annotation.Type = TokenAnnotation::TT_BinaryOperator;
      else if (Tok.Tok.is(tok::comment)) {
        StringRef Data(SourceMgr.getCharacterData(Tok.Tok.getLocation()),
                       Tok.Tok.getLength());
        if (Data.startswith("//"))
          Annotation.Type = TokenAnnotation::TT_LineComment;
        else
          Annotation.Type = TokenAnnotation::TT_BlockComment;
      }
    }
  }

  bool isUnaryOperator(unsigned Index) {
    const Token &Tok = Line.Tokens[Index].Tok;
    if (Tok.isNot(tok::minus) && Tok.isNot(tok::plus))
      return false;
    const Token &PreviousTok = Line.Tokens[Index - 1].Tok;
    if (PreviousTok.is(tok::equal) || PreviousTok.is(tok::l_paren) ||
        PreviousTok.is(tok::comma) || PreviousTok.is(tok::l_square))
      return true;
    return Annotations[Index - 1].Type == TokenAnnotation::TT_BinaryOperator;
  }

  bool isBinaryOperator(const FormatToken &Tok) {
    switch (Tok.Tok.getKind()) {
    case tok::equal:
    case tok::equalequal:
    case tok::star:
      //case tok::amp:
    case tok::plus:
    case tok::slash:
    case tok::minus:
    case tok::ampamp:
    case tok::pipe:
    case tok::pipepipe:
    case tok::percent:
      return true;
    default:
      return false;
    }
  }

  TokenAnnotation::TokenType determineStarAmpUsage(unsigned Index) {
    if (Index == Annotations.size())
      return TokenAnnotation::TT_Unknown;

    if (Index == 0 || Line.Tokens[Index - 1].Tok.is(tok::l_paren) ||
        Line.Tokens[Index - 1].Tok.is(tok::comma) ||
        Annotations[Index - 1].Type == TokenAnnotation::TT_BinaryOperator)
      return TokenAnnotation::TT_UnaryOperator;

    if (Line.Tokens[Index - 1].Tok.isLiteral() ||
        Line.Tokens[Index + 1].Tok.isLiteral())
      return TokenAnnotation::TT_BinaryOperator;

    return TokenAnnotation::TT_PointerOrReference;
  }

  bool isIfForOrWhile(Token Tok) {
    return Tok.is(tok::kw_if) || Tok.is(tok::kw_for) || Tok.is(tok::kw_while);
  }

  bool spaceRequiredBetween(Token Left, Token Right) {
    if (Left.is(tok::kw_template) && Right.is(tok::less))
      return true;
    if (Left.is(tok::arrow) || Right.is(tok::arrow))
      return false;
    if (Left.is(tok::exclaim) || Left.is(tok::tilde))
      return false;
    if (Left.is(tok::less) || Right.is(tok::greater) || Right.is(tok::less))
      return false;
    if (Left.is(tok::amp) || Left.is(tok::star))
      return Right.isLiteral() || Style.PointerAndReferenceBindToType;
    if (Right.is(tok::star) && Left.is(tok::l_paren))
      return false;
    if (Right.is(tok::amp) || Right.is(tok::star))
      return Left.isLiteral() || !Style.PointerAndReferenceBindToType;
    if (Left.is(tok::l_square) || Right.is(tok::l_square) ||
        Right.is(tok::r_square))
      return false;
    if (Left.is(tok::coloncolon) || Right.is(tok::coloncolon))
      return false;
    if (Left.is(tok::period) || Right.is(tok::period))
      return false;
    if (Left.is(tok::colon) || Right.is(tok::colon))
      return true;
    if ((Left.is(tok::plusplus) && Right.isAnyIdentifier()) ||
        (Left.isAnyIdentifier() && Right.is(tok::plusplus)) ||
        (Left.is(tok::minusminus) && Right.isAnyIdentifier()) ||
        (Left.isAnyIdentifier() && Right.is(tok::minusminus)))
      return false;
    if (Left.is(tok::l_paren))
      return false;
    if (Left.is(tok::hash))
      return false;
    if (Right.is(tok::r_paren) || Right.is(tok::semi) || Right.is(tok::comma))
      return false;
    if (Right.is(tok::l_paren)) {
      return !Left.isAnyIdentifier() || isIfForOrWhile(Left);
    }
    return true;
  }

  bool canBreakBetween(const FormatToken &Left, const FormatToken &Right) {
    if (Right.Tok.is(tok::r_paren))
      return false;
    if (isBinaryOperator(Left))
      return true;
    return Right.Tok.is(tok::colon) || Left.Tok.is(tok::comma) || Left.Tok.is(
        tok::semi) || Left.Tok.is(tok::equal) || Left.Tok.is(tok::ampamp) ||
        (Left.Tok.is(tok::l_paren) && !Right.Tok.is(tok::r_paren));
  }

  const UnwrappedLine &Line;
  FormatStyle Style;
  SourceManager &SourceMgr;
  std::vector<TokenAnnotation> Annotations;
};

class Formatter : public UnwrappedLineConsumer {
public:
  Formatter(const FormatStyle &Style, Lexer &Lex, SourceManager &SourceMgr,
            const std::vector<CharSourceRange> &Ranges)
      : Style(Style),
        Lex(Lex),
        SourceMgr(SourceMgr),
        Ranges(Ranges) {
  }

  tooling::Replacements format() {
    UnwrappedLineParser Parser(Lex, SourceMgr, *this);
    Parser.parse();
    return Replaces;
  }

private:
  virtual void formatUnwrappedLine(const UnwrappedLine &TheLine) {
    if (TheLine.Tokens.size() == 0)
      return;

    CharSourceRange LineRange =
        CharSourceRange::getTokenRange(TheLine.Tokens.front().Tok.getLocation(),
                                       TheLine.Tokens.back().Tok.getLocation());

    for (unsigned i = 0, e = Ranges.size(); i != e; ++i) {
      if (SourceMgr.isBeforeInTranslationUnit(LineRange.getEnd(),
                                              Ranges[i].getBegin()) ||
          SourceMgr.isBeforeInTranslationUnit(Ranges[i].getEnd(),
                                              LineRange.getBegin()))
        continue;

      TokenAnnotator Annotator(TheLine, Style, SourceMgr);
      Annotator.annotate();
      UnwrappedLineFormatter Formatter(Style, SourceMgr, TheLine,
                                       Annotator.getAnnotations(), Replaces);
      Formatter.format();
      return;
    }
  }

  FormatStyle Style;
  Lexer &Lex;
  SourceManager &SourceMgr;
  tooling::Replacements Replaces;
  std::vector<CharSourceRange> Ranges;
};

tooling::Replacements reformat(const FormatStyle &Style, Lexer &Lex,
                               SourceManager &SourceMgr,
                               std::vector<CharSourceRange> Ranges) {
  Formatter formatter(Style, Lex, SourceMgr, Ranges);
  return formatter.format();
}

}  // namespace format
}  // namespace clang
