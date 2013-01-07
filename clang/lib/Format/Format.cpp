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
#include "clang/Basic/OperatorPrecedence.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include <string>

namespace clang {
namespace format {

enum TokenType {
  TT_Unknown,
  TT_TemplateOpener,
  TT_TemplateCloser,
  TT_BinaryOperator,
  TT_UnaryOperator,
  TT_TrailingUnaryOperator,
  TT_OverloadedOperator,
  TT_PointerOrReference,
  TT_ConditionalExpr,
  TT_CtorInitializerColon,
  TT_LineComment,
  TT_BlockComment,
  TT_DirectorySeparator,
  TT_PureVirtualSpecifier,
  TT_ObjCMethodSpecifier
};

enum LineType {
  LT_Invalid,
  LT_Other,
  LT_PreprocessorDirective,
  LT_VirtualFunctionDecl,
  LT_ObjCMethodDecl
};

// FIXME: Move somewhere sane.
struct TokenAnnotation {
  TokenType Type;

  bool SpaceRequiredBefore;
  bool CanBreakBefore;
  bool MustBreakBefore;

  bool ClosesTemplateDeclaration;
};

static prec::Level getPrecedence(const FormatToken &Tok) {
  return getBinOpPrecedence(Tok.Tok.getKind(), true, true);
}

using llvm::MutableArrayRef;

FormatStyle getLLVMStyle() {
  FormatStyle LLVMStyle;
  LLVMStyle.ColumnLimit = 80;
  LLVMStyle.MaxEmptyLinesToKeep = 1;
  LLVMStyle.PointerAndReferenceBindToType = false;
  LLVMStyle.AccessModifierOffset = -2;
  LLVMStyle.SplitTemplateClosingGreater = true;
  LLVMStyle.IndentCaseLabels = false;
  LLVMStyle.SpacesBeforeTrailingComments = 1;
  return LLVMStyle;
}

FormatStyle getGoogleStyle() {
  FormatStyle GoogleStyle;
  GoogleStyle.ColumnLimit = 80;
  GoogleStyle.MaxEmptyLinesToKeep = 1;
  GoogleStyle.PointerAndReferenceBindToType = true;
  GoogleStyle.AccessModifierOffset = -1;
  GoogleStyle.SplitTemplateClosingGreater = false;
  GoogleStyle.IndentCaseLabels = true;
  GoogleStyle.SpacesBeforeTrailingComments = 2;
  return GoogleStyle;
}

struct OptimizationParameters {
  unsigned PenaltyIndentLevel;
  unsigned PenaltyLevelDecrease;
};

class UnwrappedLineFormatter {
public:
  UnwrappedLineFormatter(const FormatStyle &Style, SourceManager &SourceMgr,
                         const UnwrappedLine &Line,
                         unsigned PreviousEndOfLineColumn,
                         LineType CurrentLineType,
                         const std::vector<TokenAnnotation> &Annotations,
                         tooling::Replacements &Replaces, bool StructuralError)
      : Style(Style), SourceMgr(SourceMgr), Line(Line),
        PreviousEndOfLineColumn(PreviousEndOfLineColumn),
        CurrentLineType(CurrentLineType), Annotations(Annotations),
        Replaces(Replaces), StructuralError(StructuralError) {
    Parameters.PenaltyIndentLevel = 15;
    Parameters.PenaltyLevelDecrease = 30;
  }

  /// \brief Formats an \c UnwrappedLine.
  ///
  /// \returns The column after the last token in the last line of the
  /// \c UnwrappedLine.
  unsigned format() {
    // Format first token and initialize indent.
    unsigned Indent = formatFirstToken();

    // Initialize state dependent on indent.
    IndentState State;
    State.Column = Indent;
    State.ConsumedTokens = 0;
    State.Indent.push_back(Indent + 4);
    State.LastSpace.push_back(Indent);
    State.FirstLessLess.push_back(0);
    State.ForLoopVariablePos = 0;
    State.LineContainsContinuedForLoopSection = false;
    State.StartOfLineLevel = 1;

    // The first token has already been indented and thus consumed.
    moveStateToNextToken(State);

    // Check whether the UnwrappedLine can be put onto a single line. If so,
    // this is bound to be the optimal solution (by definition) and we don't
    // need to analyze the entire solution space.
    unsigned Columns = State.Column;
    bool FitsOnALine = true;
    for (unsigned i = 1, n = Line.Tokens.size(); i != n; ++i) {
      Columns += (Annotations[i].SpaceRequiredBefore ? 1 : 0) +
                 Line.Tokens[i].TokenLength;
      // A special case for the colon of a constructor initializer as this only
      // needs to be put on a new line if the line needs to be split.
      if (Columns > Style.ColumnLimit - (Line.InPPDirective ? 1 : 0) ||
          (Annotations[i].MustBreakBefore &&
           Annotations[i].Type != TT_CtorInitializerColon)) {
        FitsOnALine = false;
        break;
      }
    }

    // Start iterating at 1 as we have correctly formatted of Token #0 above.
    for (unsigned i = 1, n = Line.Tokens.size(); i != n; ++i) {
      if (FitsOnALine) {
        addTokenToState(false, false, State);
      } else {
        unsigned NoBreak = calcPenalty(State, false, UINT_MAX);
        unsigned Break = calcPenalty(State, true, NoBreak);
        addTokenToState(Break < NoBreak, false, State);
      }
    }
    return State.Column;
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

    /// \brief The parenthesis level of the first token on the current line.
    unsigned StartOfLineLevel;

    /// \brief The position to which a specific parenthesis level needs to be
    /// indented.
    std::vector<unsigned> Indent;

    /// \brief The position of the last space on each level.
    ///
    /// Used e.g. to break like:
    /// functionCall(Parameter, otherCall(
    ///                             OtherParameter));
    std::vector<unsigned> LastSpace;

    /// \brief The position the first "<<" operator encountered on each level.
    ///
    /// Used to align "<<" operators. 0 if no such operator has been encountered
    /// on a level.
    std::vector<unsigned> FirstLessLess;

    /// \brief The column of the first variable in a for-loop declaration.
    ///
    /// Used to align the second variable if necessary.
    unsigned ForLoopVariablePos;

    /// \brief \c true if this line contains a continued for-loop section.
    bool LineContainsContinuedForLoopSection;

    /// \brief Comparison operator to be able to used \c IndentState in \c map.
    bool operator<(const IndentState &Other) const {
      if (Other.ConsumedTokens != ConsumedTokens)
        return Other.ConsumedTokens > ConsumedTokens;
      if (Other.Column != Column)
        return Other.Column > Column;
      if (Other.StartOfLineLevel != StartOfLineLevel)
        return Other.StartOfLineLevel > StartOfLineLevel;
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
      if (Other.FirstLessLess.size() != FirstLessLess.size())
        return Other.FirstLessLess.size() > FirstLessLess.size();
      for (int i = 0, e = FirstLessLess.size(); i != e; ++i) {
        if (Other.FirstLessLess[i] != FirstLessLess[i])
          return Other.FirstLessLess[i] > FirstLessLess[i];
      }
      if (Other.ForLoopVariablePos != ForLoopVariablePos)
        return Other.ForLoopVariablePos < ForLoopVariablePos;
      if (Other.LineContainsContinuedForLoopSection !=
          LineContainsContinuedForLoopSection)
        return LineContainsContinuedForLoopSection;
      return false;
    }
  };

  /// \brief Appends the next token to \p State and updates information
  /// necessary for indentation.
  ///
  /// Puts the token on the current line if \p Newline is \c true and adds a
  /// line break and necessary indentation otherwise.
  ///
  /// If \p DryRun is \c false, also creates and stores the required
  /// \c Replacement.
  void addTokenToState(bool Newline, bool DryRun, IndentState &State) {
    unsigned Index = State.ConsumedTokens;
    const FormatToken &Current = Line.Tokens[Index];
    const FormatToken &Previous = Line.Tokens[Index - 1];
    unsigned ParenLevel = State.Indent.size() - 1;

    if (Newline) {
      unsigned WhitespaceStartColumn = State.Column;
      if (Current.Tok.is(tok::string_literal) &&
          Previous.Tok.is(tok::string_literal)) {
        State.Column = State.Column - Previous.TokenLength;
      } else if (Current.Tok.is(tok::lessless) &&
                 State.FirstLessLess[ParenLevel] != 0) {
        State.Column = State.FirstLessLess[ParenLevel];
      } else if (ParenLevel != 0 &&
                 (Previous.Tok.is(tok::equal) || Current.Tok.is(tok::arrow) ||
                  Current.Tok.is(tok::period))) {
        // Indent and extra 4 spaces after '=' as it continues an expression.
        // Don't do that on the top level, as we already indent 4 there.
        State.Column = State.Indent[ParenLevel] + 4;
      } else if (Line.Tokens[0].Tok.is(tok::kw_for) &&
                 Previous.Tok.is(tok::comma)) {
        State.Column = State.ForLoopVariablePos;
      } else if (Annotations[Index - 1].ClosesTemplateDeclaration) {
        State.Column = State.Indent[ParenLevel] - 4;
      } else {
        State.Column = State.Indent[ParenLevel];
      }

      State.StartOfLineLevel = ParenLevel + 1;

      if (Line.Tokens[0].Tok.is(tok::kw_for))
        State.LineContainsContinuedForLoopSection =
            Previous.Tok.isNot(tok::semi);

      if (!DryRun) {
        if (!Line.InPPDirective)
          replaceWhitespace(Current, 1, State.Column);
        else
          replacePPWhitespace(Current, 1, State.Column, WhitespaceStartColumn);
      }

      State.LastSpace[ParenLevel] = State.Indent[ParenLevel];
      if (Current.Tok.is(tok::colon) && CurrentLineType != LT_ObjCMethodDecl &&
          Annotations[Index].Type != TT_ConditionalExpr)
        State.Indent[ParenLevel] += 2;
    } else {
      if (Current.Tok.is(tok::equal) && Line.Tokens[0].Tok.is(tok::kw_for))
        State.ForLoopVariablePos = State.Column - Previous.TokenLength;

      unsigned Spaces = Annotations[Index].SpaceRequiredBefore ? 1 : 0;
      if (Annotations[Index].Type == TT_LineComment)
        Spaces = Style.SpacesBeforeTrailingComments;

      if (!DryRun)
        replaceWhitespace(Current, 0, Spaces);

      // FIXME: Look into using this alignment at other ParenLevels.
      if (ParenLevel == 0 && (getPrecedence(Previous) == prec::Assignment ||
                              Previous.Tok.is(tok::kw_return)))
        State.Indent[ParenLevel] = State.Column + Spaces;
      if (Previous.Tok.is(tok::l_paren) ||
          Annotations[Index - 1].Type == TT_TemplateOpener)
        State.Indent[ParenLevel] = State.Column;

      // Top-level spaces are exempt as that mostly leads to better results.
      State.Column += Spaces;
      if (Spaces > 0 && ParenLevel != 0)
        State.LastSpace[ParenLevel] = State.Column;
    }
    moveStateToNextToken(State);
  }

  /// \brief Mark the next token as consumed in \p State and modify its stacks
  /// accordingly.
  void moveStateToNextToken(IndentState &State) {
    unsigned Index = State.ConsumedTokens;
    const FormatToken &Current = Line.Tokens[Index];
    unsigned ParenLevel = State.Indent.size() - 1;

    if (Current.Tok.is(tok::lessless) && State.FirstLessLess[ParenLevel] == 0)
      State.FirstLessLess[ParenLevel] = State.Column;

    State.Column += Current.TokenLength;

    // If we encounter an opening (, [, { or <, we add a level to our stacks to
    // prepare for the following tokens.
    if (Current.Tok.is(tok::l_paren) || Current.Tok.is(tok::l_square) ||
        Current.Tok.is(tok::l_brace) ||
        Annotations[Index].Type == TT_TemplateOpener) {
      State.Indent.push_back(4 + State.LastSpace.back());
      State.LastSpace.push_back(State.LastSpace.back());
      State.FirstLessLess.push_back(0);
    }

    // If we encounter a closing ), ], } or >, we can remove a level from our
    // stacks.
    if (Current.Tok.is(tok::r_paren) || Current.Tok.is(tok::r_square) ||
        (Current.Tok.is(tok::r_brace) && State.ConsumedTokens > 0) ||
        Annotations[Index].Type == TT_TemplateCloser) {
      State.Indent.pop_back();
      State.LastSpace.pop_back();
      State.FirstLessLess.pop_back();
    }

    ++State.ConsumedTokens;
  }

  /// \brief Calculate the panelty for splitting after the token at \p Index.
  unsigned splitPenalty(unsigned Index) {
    assert(Index < Line.Tokens.size() &&
           "Tried to calculate penalty for splitting after the last token");
    const FormatToken &Left = Line.Tokens[Index];
    const FormatToken &Right = Line.Tokens[Index + 1];

    // In for-loops, prefer breaking at ',' and ';'.
    if (Line.Tokens[0].Tok.is(tok::kw_for) &&
        (Left.Tok.isNot(tok::comma) && Left.Tok.isNot(tok::semi)))
      return 20;

    if (Left.Tok.is(tok::semi) || Left.Tok.is(tok::comma) ||
        Annotations[Index].ClosesTemplateDeclaration)
      return 0;
    if (Left.Tok.is(tok::l_paren))
      return 20;

    prec::Level Level = getPrecedence(Line.Tokens[Index]);
    if (Level != prec::Unknown)
      return Level;

    if (Right.Tok.is(tok::arrow) || Right.Tok.is(tok::period))
      return 150;

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
    if (!NewLine && Line.Tokens[State.ConsumedTokens - 1].Tok.is(tok::semi) &&
        State.LineContainsContinuedForLoopSection)
      return UINT_MAX;

    unsigned CurrentPenalty = 0;
    if (NewLine) {
      CurrentPenalty += Parameters.PenaltyIndentLevel * State.Indent.size() +
                        splitPenalty(State.ConsumedTokens - 1);
    } else {
      if (State.Indent.size() < State.StartOfLineLevel)
        CurrentPenalty += Parameters.PenaltyLevelDecrease *
                          (State.StartOfLineLevel - State.Indent.size());
    }

    addTokenToState(NewLine, true, State);

    // Exceeding column limit is bad.
    if (State.Column > Style.ColumnLimit - (Line.InPPDirective ? 1 : 0))
      return UINT_MAX;

    if (StopAt <= CurrentPenalty)
      return UINT_MAX;
    StopAt -= CurrentPenalty;

    StateMap::iterator I = Memory.find(State);
    if (I != Memory.end()) {
      // If this state has already been examined, we can safely return the
      // previous result if we
      // - have not hit the optimatization (and thus returned UINT_MAX) OR
      // - are now computing for a smaller or equal StopAt.
      unsigned SavedResult = I->second.first;
      unsigned SavedStopAt = I->second.second;
      if (SavedResult != UINT_MAX)
        return SavedResult + CurrentPenalty;
      else if (StopAt <= SavedStopAt)
        return UINT_MAX;
    }

    unsigned NoBreak = calcPenalty(State, false, StopAt);
    unsigned WithBreak = calcPenalty(State, true, std::min(StopAt, NoBreak));
    unsigned Result = std::min(NoBreak, WithBreak);

    // We have to store 'Result' without adding 'CurrentPenalty' as the latter
    // can depend on 'NewLine'.
    Memory[State] = std::pair<unsigned, unsigned>(Result, StopAt);

    return Result == UINT_MAX ? UINT_MAX : Result + CurrentPenalty;
  }

  /// \brief Replaces the whitespace in front of \p Tok. Only call once for
  /// each \c FormatToken.
  void replaceWhitespace(const FormatToken &Tok, unsigned NewLines,
                         unsigned Spaces) {
    Replaces.insert(tooling::Replacement(
        SourceMgr, Tok.WhiteSpaceStart, Tok.WhiteSpaceLength,
        std::string(NewLines, '\n') + std::string(Spaces, ' ')));
  }

  /// \brief Like \c replaceWhitespace, but additionally adds right-aligned
  /// backslashes to escape newlines inside a preprocessor directive.
  ///
  /// This function and \c replaceWhitespace have the same behavior if
  /// \c Newlines == 0.
  void replacePPWhitespace(const FormatToken &Tok, unsigned NewLines,
                           unsigned Spaces, unsigned WhitespaceStartColumn) {
    std::string NewLineText;
    if (NewLines > 0) {
      unsigned Offset =
          std::min<int>(Style.ColumnLimit - 1, WhitespaceStartColumn);
      for (unsigned i = 0; i < NewLines; ++i) {
        NewLineText += std::string(Style.ColumnLimit - Offset - 1, ' ');
        NewLineText += "\\\n";
        Offset = 0;
      }
    }
    Replaces.insert(tooling::Replacement(
        SourceMgr, Tok.WhiteSpaceStart, Tok.WhiteSpaceLength,
        NewLineText + std::string(Spaces, ' ')));
  }

  /// \brief Add a new line and the required indent before the first Token
  /// of the \c UnwrappedLine if there was no structural parsing error.
  /// Returns the indent level of the \c UnwrappedLine.
  unsigned formatFirstToken() {
    const FormatToken &Token = Line.Tokens[0];
    if (!Token.WhiteSpaceStart.isValid() || StructuralError)
      return SourceMgr.getSpellingColumnNumber(Token.Tok.getLocation()) - 1;

    unsigned Newlines =
        std::min(Token.NewlinesBefore, Style.MaxEmptyLinesToKeep + 1);
    if (Newlines == 0 && !Token.IsFirst)
      Newlines = 1;
    unsigned Indent = Line.Level * 2;
    if ((Token.Tok.is(tok::kw_public) || Token.Tok.is(tok::kw_protected) ||
         Token.Tok.is(tok::kw_private)) &&
        static_cast<int>(Indent) + Style.AccessModifierOffset >= 0)
      Indent += Style.AccessModifierOffset;
    if (!Line.InPPDirective || Token.HasUnescapedNewline)
      replaceWhitespace(Token, Newlines, Indent);
    else
      replacePPWhitespace(Token, Newlines, Indent, PreviousEndOfLineColumn);
    return Indent;
  }

  FormatStyle Style;
  SourceManager &SourceMgr;
  const UnwrappedLine &Line;
  const unsigned PreviousEndOfLineColumn;
  const LineType CurrentLineType;
  const std::vector<TokenAnnotation> &Annotations;
  tooling::Replacements &Replaces;
  bool StructuralError;

  // A map from an indent state to a pair (Result, Used-StopAt).
  typedef std::map<IndentState, std::pair<unsigned, unsigned> > StateMap;
  StateMap Memory;

  OptimizationParameters Parameters;
};

/// \brief Determines extra information about the tokens comprising an
/// \c UnwrappedLine.
class TokenAnnotator {
public:
  TokenAnnotator(const UnwrappedLine &Line, const FormatStyle &Style,
                 SourceManager &SourceMgr, Lexer &Lex)
      : Line(Line), Style(Style), SourceMgr(SourceMgr), Lex(Lex) {
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
        : Tokens(Tokens), Annotations(Annotations), Index(0),
          KeywordVirtualFound(false) {
    }

    bool parseAngle() {
      while (Index < Tokens.size()) {
        if (Tokens[Index].Tok.is(tok::greater)) {
          Annotations[Index].Type = TT_TemplateCloser;
          next();
          return true;
        }
        if (Tokens[Index].Tok.is(tok::r_paren) ||
            Tokens[Index].Tok.is(tok::r_square) ||
            Tokens[Index].Tok.is(tok::r_brace))
          return false;
        if (Tokens[Index].Tok.is(tok::pipepipe) ||
            Tokens[Index].Tok.is(tok::ampamp) ||
            Tokens[Index].Tok.is(tok::question) ||
            Tokens[Index].Tok.is(tok::colon))
          return false;
        if (!consumeToken())
          return false;
      }
      return false;
    }

    bool parseParens() {
      while (Index < Tokens.size()) {
        if (Tokens[Index].Tok.is(tok::r_paren)) {
          next();
          return true;
        }
        if (Tokens[Index].Tok.is(tok::r_square) ||
            Tokens[Index].Tok.is(tok::r_brace))
          return false;
        if (!consumeToken())
          return false;
      }
      return false;
    }

    bool parseSquare() {
      while (Index < Tokens.size()) {
        if (Tokens[Index].Tok.is(tok::r_square)) {
          next();
          return true;
        }
        if (Tokens[Index].Tok.is(tok::r_paren) ||
            Tokens[Index].Tok.is(tok::r_brace))
          return false;
        if (!consumeToken())
          return false;
      }
      return false;
    }

    bool parseConditional() {
      while (Index < Tokens.size()) {
        if (Tokens[Index].Tok.is(tok::colon)) {
          Annotations[Index].Type = TT_ConditionalExpr;
          next();
          return true;
        }
        if (!consumeToken())
          return false;
      }
      return false;
    }

    bool parseTemplateDeclaration() {
      if (Index < Tokens.size() && Tokens[Index].Tok.is(tok::less)) {
        Annotations[Index].Type = TT_TemplateOpener;
        next();
        if (!parseAngle())
          return false;
        Annotations[Index - 1].ClosesTemplateDeclaration = true;
        parseLine();
        return true;
      }
      return false;
    }

    bool consumeToken() {
      unsigned CurrentIndex = Index;
      next();
      switch (Tokens[CurrentIndex].Tok.getKind()) {
      case tok::l_paren:
        if (!parseParens())
          return false;
        if (Index < Tokens.size() && Tokens[Index].Tok.is(tok::colon)) {
          Annotations[Index].Type = TT_CtorInitializerColon;
          next();
        }
        break;
      case tok::l_square:
        if (!parseSquare())
          return false;
        break;
      case tok::less:
        if (parseAngle())
          Annotations[CurrentIndex].Type = TT_TemplateOpener;
        else {
          Annotations[CurrentIndex].Type = TT_BinaryOperator;
          Index = CurrentIndex + 1;
        }
        break;
      case tok::r_paren:
      case tok::r_square:
        return false;
      case tok::greater:
        Annotations[CurrentIndex].Type = TT_BinaryOperator;
        break;
      case tok::kw_operator:
        if (Tokens[Index].Tok.is(tok::l_paren)) {
          Annotations[Index].Type = TT_OverloadedOperator;
          next();
          if (Index < Tokens.size() && Tokens[Index].Tok.is(tok::r_paren)) {
            Annotations[Index].Type = TT_OverloadedOperator;
            next();
          }
        } else {
          while (Index < Tokens.size() && !Tokens[Index].Tok.is(tok::l_paren)) {
            Annotations[Index].Type = TT_OverloadedOperator;
            next();
          }
        }
        break;
      case tok::question:
        parseConditional();
        break;
      case tok::kw_template:
        parseTemplateDeclaration();
        break;
      default:
        break;
      }
      return true;
    }

    void parseIncludeDirective() {
      while (Index < Tokens.size()) {
        if (Tokens[Index].Tok.is(tok::slash))
          Annotations[Index].Type = TT_DirectorySeparator;
        else if (Tokens[Index].Tok.is(tok::less))
          Annotations[Index].Type = TT_TemplateOpener;
        else if (Tokens[Index].Tok.is(tok::greater))
          Annotations[Index].Type = TT_TemplateCloser;
        next();
      }
    }

    void parsePreprocessorDirective() {
      next();
      if (Index >= Tokens.size())
        return;
      // Hashes in the middle of a line can lead to any strange token
      // sequence.
      if (Tokens[Index].Tok.getIdentifierInfo() == NULL)
        return;
      switch (Tokens[Index].Tok.getIdentifierInfo()->getPPKeywordID()) {
      case tok::pp_include:
      case tok::pp_import:
        parseIncludeDirective();
        break;
      default:
        break;
      }
    }

    LineType parseLine() {
      if (Tokens[Index].Tok.is(tok::hash)) {
        parsePreprocessorDirective();
        return LT_PreprocessorDirective;
      }
      while (Index < Tokens.size()) {
        if (Tokens[Index].Tok.is(tok::kw_virtual))
          KeywordVirtualFound = true;
        if (!consumeToken())
          return LT_Invalid;
      }
      if (KeywordVirtualFound)
        return LT_VirtualFunctionDecl;
      return LT_Other;
    }

    void next() {
      ++Index;
    }

  private:
    const SmallVector<FormatToken, 16> &Tokens;
    std::vector<TokenAnnotation> &Annotations;
    unsigned Index;
    bool KeywordVirtualFound;
  };

  bool annotate() {
    if (Line.Tokens.size() == 0)
      return true;

    Annotations.clear();
    for (int i = 0, e = Line.Tokens.size(); i != e; ++i) {
      Annotations.push_back(TokenAnnotation());
    }

    AnnotatingParser Parser(Line.Tokens, Annotations);
    CurrentLineType = Parser.parseLine();
    if (CurrentLineType == LT_Invalid)
      return false;

    determineTokenTypes();

    if (Annotations[0].Type == TT_ObjCMethodSpecifier)
      CurrentLineType = LT_ObjCMethodDecl;

    for (int i = 1, e = Line.Tokens.size(); i != e; ++i) {
      TokenAnnotation &Annotation = Annotations[i];

      Annotation.CanBreakBefore = canBreakBefore(i);

      if (Annotation.Type == TT_CtorInitializerColon) {
        Annotation.MustBreakBefore = true;
        Annotation.SpaceRequiredBefore = true;
      } else if (Annotation.Type == TT_OverloadedOperator) {
        Annotation.SpaceRequiredBefore =
            Line.Tokens[i].Tok.is(tok::identifier) ||
            Line.Tokens[i].Tok.is(tok::kw_new) ||
            Line.Tokens[i].Tok.is(tok::kw_delete);
      } else if (Annotations[i - 1].Type == TT_OverloadedOperator) {
        Annotation.SpaceRequiredBefore = false;
      } else if (CurrentLineType == LT_ObjCMethodDecl &&
                 Line.Tokens[i].Tok.is(tok::identifier) && (i != e - 1) &&
                 Line.Tokens[i + 1].Tok.is(tok::colon) &&
                 Line.Tokens[i - 1].Tok.is(tok::identifier)) {
        Annotation.CanBreakBefore = true;
        Annotation.SpaceRequiredBefore = true;
      } else if (CurrentLineType == LT_ObjCMethodDecl &&
                 Line.Tokens[i].Tok.is(tok::identifier) &&
                 Line.Tokens[i - 1].Tok.is(tok::l_paren) &&
                 Line.Tokens[i - 2].Tok.is(tok::colon)) {
        // Don't break this identifier as ':' or identifier
        // before it will break.
        Annotation.CanBreakBefore = false;
      } else if (Line.Tokens[i].Tok.is(tok::at) &&
                 Line.Tokens[i - 2].Tok.is(tok::at)) {
        // Don't put two objc's '@' on the same line. This could happen,
        // as in, @optional @property ...
        Annotation.MustBreakBefore = true;
      } else if (Line.Tokens[i].Tok.is(tok::colon)) {
        Annotation.SpaceRequiredBefore =
            Line.Tokens[0].Tok.isNot(tok::kw_case) &&
            CurrentLineType != LT_ObjCMethodDecl && (i != e - 1);
        // Don't break at ':' if identifier before it can beak.
        if (CurrentLineType == LT_ObjCMethodDecl &&
            Line.Tokens[i - 1].Tok.is(tok::identifier) &&
            Annotations[i - 1].CanBreakBefore)
          Annotation.CanBreakBefore = false;
      } else if (Annotations[i - 1].Type == TT_ObjCMethodSpecifier) {
        Annotation.SpaceRequiredBefore = true;
      } else if (Annotations[i - 1].Type == TT_UnaryOperator) {
        Annotation.SpaceRequiredBefore = false;
      } else if (Annotation.Type == TT_UnaryOperator) {
        Annotation.SpaceRequiredBefore =
            Line.Tokens[i - 1].Tok.isNot(tok::l_paren) &&
            Line.Tokens[i - 1].Tok.isNot(tok::l_square);
      } else if (Line.Tokens[i - 1].Tok.is(tok::greater) &&
                 Line.Tokens[i].Tok.is(tok::greater)) {
        if (Annotation.Type == TT_TemplateCloser &&
            Annotations[i - 1].Type == TT_TemplateCloser)
          Annotation.SpaceRequiredBefore = Style.SplitTemplateClosingGreater;
        else
          Annotation.SpaceRequiredBefore = false;
      } else if (Annotation.Type == TT_DirectorySeparator ||
                 Annotations[i - 1].Type == TT_DirectorySeparator) {
        Annotation.SpaceRequiredBefore = false;
      } else if (Annotation.Type == TT_BinaryOperator ||
                 Annotations[i - 1].Type == TT_BinaryOperator) {
        Annotation.SpaceRequiredBefore = true;
      } else if (Annotations[i - 1].Type == TT_TemplateCloser &&
                 Line.Tokens[i].Tok.is(tok::l_paren)) {
        Annotation.SpaceRequiredBefore = false;
      } else if (Line.Tokens[i].Tok.is(tok::less) &&
                 Line.Tokens[0].Tok.is(tok::hash)) {
        Annotation.SpaceRequiredBefore = true;
      } else if (CurrentLineType == LT_ObjCMethodDecl &&
                 Line.Tokens[i - 1].Tok.is(tok::r_paren) &&
                 Line.Tokens[i].Tok.is(tok::identifier)) {
        // Don't space between ')' and <id>
        Annotation.SpaceRequiredBefore = false;
      } else if (CurrentLineType == LT_ObjCMethodDecl &&
                 Line.Tokens[i - 1].Tok.is(tok::colon) &&
                 Line.Tokens[i].Tok.is(tok::l_paren)) {
        // Don't space between ':' and '('
        Annotation.SpaceRequiredBefore = false;
      } else if (Annotation.Type == TT_TrailingUnaryOperator) {
        Annotation.SpaceRequiredBefore = false;
      } else {
        Annotation.SpaceRequiredBefore =
            spaceRequiredBetween(Line.Tokens[i - 1].Tok, Line.Tokens[i].Tok);
      }

      if (Annotations[i - 1].Type == TT_LineComment ||
          (Line.Tokens[i].Tok.is(tok::string_literal) &&
           Line.Tokens[i - 1].Tok.is(tok::string_literal))) {
        Annotation.MustBreakBefore = true;
      }

      if (Annotation.MustBreakBefore)
        Annotation.CanBreakBefore = true;
    }
    return true;
  }

  LineType getLineType() {
    return CurrentLineType;
  }

  const std::vector<TokenAnnotation> &getAnnotations() {
    return Annotations;
  }

private:
  void determineTokenTypes() {
    bool IsRHS = false;
    for (int i = 0, e = Line.Tokens.size(); i != e; ++i) {
      TokenAnnotation &Annotation = Annotations[i];
      const FormatToken &Tok = Line.Tokens[i];

      if (getPrecedence(Tok) == prec::Assignment)
        IsRHS = true;
      else if (Tok.Tok.is(tok::kw_return))
        IsRHS = true;

      if (Annotation.Type != TT_Unknown)
        continue;

      if (Tok.Tok.is(tok::star) || Tok.Tok.is(tok::amp)) {
        Annotation.Type = determineStarAmpUsage(i, IsRHS);
      } else if (Tok.Tok.is(tok::minus) || Tok.Tok.is(tok::plus)) {
        Annotation.Type = determinePlusMinusUsage(i);
      } else if (Tok.Tok.is(tok::minusminus) || Tok.Tok.is(tok::plusplus)) {
        Annotation.Type = determineIncrementUsage(i);
      } else if (Tok.Tok.is(tok::exclaim)) {
        Annotation.Type = TT_UnaryOperator;
      } else if (isBinaryOperator(Line.Tokens[i])) {
        Annotation.Type = TT_BinaryOperator;
      } else if (Tok.Tok.is(tok::comment)) {
        std::string Data(
            Lexer::getSpelling(Tok.Tok, SourceMgr, Lex.getLangOpts()));
        if (StringRef(Data).startswith("//"))
          Annotation.Type = TT_LineComment;
        else
          Annotation.Type = TT_BlockComment;
      }
    }
  }

  bool isBinaryOperator(const FormatToken &Tok) {
    // Comma is a binary operator, but does not behave as such wrt. formatting.
    return getPrecedence(Tok) > prec::Comma;
  }

  TokenType determineStarAmpUsage(unsigned Index, bool IsRHS) {
    if (Index == 0)
      return TT_UnaryOperator;
    if (Index == Annotations.size())
      return TT_Unknown;
    const FormatToken &PrevToken = Line.Tokens[Index - 1];
    const FormatToken &NextToken = Line.Tokens[Index + 1];

    if (PrevToken.Tok.is(tok::l_paren) || PrevToken.Tok.is(tok::l_square) ||
        PrevToken.Tok.is(tok::comma) || PrevToken.Tok.is(tok::kw_return) ||
        PrevToken.Tok.is(tok::colon) ||
        Annotations[Index - 1].Type == TT_BinaryOperator)
      return TT_UnaryOperator;

    if (PrevToken.Tok.isLiteral() || NextToken.Tok.isLiteral() ||
        NextToken.Tok.is(tok::plus) || NextToken.Tok.is(tok::minus) ||
        NextToken.Tok.is(tok::plusplus) || NextToken.Tok.is(tok::minusminus) ||
        NextToken.Tok.is(tok::tilde) || NextToken.Tok.is(tok::exclaim) ||
        NextToken.Tok.is(tok::kw_alignof) || NextToken.Tok.is(tok::kw_sizeof))
      return TT_BinaryOperator;

    if (NextToken.Tok.is(tok::comma) || NextToken.Tok.is(tok::r_paren) ||
        NextToken.Tok.is(tok::greater))
      return TT_PointerOrReference;

    // It is very unlikely that we are going to find a pointer or reference type
    // definition on the RHS of an assignment.
    if (IsRHS)
      return TT_BinaryOperator;

    return TT_PointerOrReference;
  }

  TokenType determinePlusMinusUsage(unsigned Index) {
    // At the start of the line, +/- specific ObjectiveC method declarations.
    if (Index == 0)
      return TT_ObjCMethodSpecifier;

    // Use heuristics to recognize unary operators.
    const Token &PreviousTok = Line.Tokens[Index - 1].Tok;
    if (PreviousTok.is(tok::equal) || PreviousTok.is(tok::l_paren) ||
        PreviousTok.is(tok::comma) || PreviousTok.is(tok::l_square) ||
        PreviousTok.is(tok::question) || PreviousTok.is(tok::colon) ||
        PreviousTok.is(tok::kw_return) || PreviousTok.is(tok::kw_case))
      return TT_UnaryOperator;

    // There can't be to consecutive binary operators.
    if (Annotations[Index - 1].Type == TT_BinaryOperator)
      return TT_UnaryOperator;

    // Fall back to marking the token as binary operator.
    return TT_BinaryOperator;
  }

  /// \brief Determine whether ++/-- are pre- or post-increments/-decrements.
  TokenType determineIncrementUsage(unsigned Index) {
    if (Index != 0 && Line.Tokens[Index - 1].Tok.is(tok::identifier))
      return TT_TrailingUnaryOperator;

    return TT_UnaryOperator;
  }

  bool spaceRequiredBetween(Token Left, Token Right) {
    if (Right.is(tok::r_paren) || Right.is(tok::semi) || Right.is(tok::comma))
      return false;
    if (Left.is(tok::kw_template) && Right.is(tok::less))
      return true;
    if (Left.is(tok::arrow) || Right.is(tok::arrow))
      return false;
    if (Left.is(tok::exclaim) || Left.is(tok::tilde))
      return false;
    if (Left.is(tok::at) && Right.is(tok::identifier))
      return false;
    if (Left.is(tok::less) || Right.is(tok::greater) || Right.is(tok::less))
      return false;
    if (Right.is(tok::amp) || Right.is(tok::star))
      return Left.isLiteral() ||
             (Left.isNot(tok::star) && Left.isNot(tok::amp) &&
              !Style.PointerAndReferenceBindToType);
    if (Left.is(tok::amp) || Left.is(tok::star))
      return Right.isLiteral() || Style.PointerAndReferenceBindToType;
    if (Right.is(tok::star) && Left.is(tok::l_paren))
      return false;
    if (Left.is(tok::l_square) || Right.is(tok::l_square) ||
        Right.is(tok::r_square))
      return false;
    if (Left.is(tok::coloncolon) ||
        (Right.is(tok::coloncolon) &&
         (Left.is(tok::identifier) || Left.is(tok::greater))))
      return false;
    if (Left.is(tok::period) || Right.is(tok::period))
      return false;
    if (Left.is(tok::colon) || Right.is(tok::colon))
      return true;
    if (Left.is(tok::l_paren))
      return false;
    if (Left.is(tok::hash))
      return false;
    if (Right.is(tok::l_paren)) {
      return Left.is(tok::kw_if) || Left.is(tok::kw_for) ||
             Left.is(tok::kw_while) || Left.is(tok::kw_switch) ||
             (Left.isNot(tok::identifier) && Left.isNot(tok::kw_sizeof) &&
              Left.isNot(tok::kw_typeof) && Left.isNot(tok::kw_alignof));
    }
    return true;
  }

  bool canBreakBefore(unsigned i) {
    if (Annotations[i - 1].ClosesTemplateDeclaration)
      return true;
    if (Annotations[i - 1].Type == TT_PointerOrReference ||
        Annotations[i - 1].Type == TT_TemplateCloser ||
        Annotations[i].Type == TT_ConditionalExpr) {
      return false;
    }
    const FormatToken &Left = Line.Tokens[i - 1];
    const FormatToken &Right = Line.Tokens[i];
    if (Left.Tok.is(tok::equal) && CurrentLineType == LT_VirtualFunctionDecl)
      return false;

    if (Right.Tok.is(tok::r_paren) || Right.Tok.is(tok::l_brace) ||
        Right.Tok.is(tok::comment) || Right.Tok.is(tok::greater))
      return false;
    return (isBinaryOperator(Left) && Left.Tok.isNot(tok::lessless)) ||
           Left.Tok.is(tok::comma) || Right.Tok.is(tok::lessless) ||
           Right.Tok.is(tok::arrow) || Right.Tok.is(tok::period) ||
           Right.Tok.is(tok::colon) || Left.Tok.is(tok::semi) ||
           Left.Tok.is(tok::l_brace) ||
           (Left.Tok.is(tok::l_paren) && !Right.Tok.is(tok::r_paren));
  }

  const UnwrappedLine &Line;
  FormatStyle Style;
  SourceManager &SourceMgr;
  Lexer &Lex;
  LineType CurrentLineType;
  std::vector<TokenAnnotation> Annotations;
};

class LexerBasedFormatTokenSource : public FormatTokenSource {
public:
  LexerBasedFormatTokenSource(Lexer &Lex, SourceManager &SourceMgr)
      : GreaterStashed(false), Lex(Lex), SourceMgr(SourceMgr),
        IdentTable(Lex.getLangOpts()) {
    Lex.SetKeepWhitespaceMode(true);
  }

  virtual FormatToken getNextToken() {
    if (GreaterStashed) {
      FormatTok.NewlinesBefore = 0;
      FormatTok.WhiteSpaceStart =
          FormatTok.Tok.getLocation().getLocWithOffset(1);
      FormatTok.WhiteSpaceLength = 0;
      GreaterStashed = false;
      return FormatTok;
    }

    FormatTok = FormatToken();
    Lex.LexFromRawLexer(FormatTok.Tok);
    StringRef Text = rawTokenText(FormatTok.Tok);
    FormatTok.WhiteSpaceStart = FormatTok.Tok.getLocation();
    if (SourceMgr.getFileOffset(FormatTok.WhiteSpaceStart) == 0)
      FormatTok.IsFirst = true;

    // Consume and record whitespace until we find a significant token.
    while (FormatTok.Tok.is(tok::unknown)) {
      FormatTok.NewlinesBefore += Text.count('\n');
      FormatTok.HasUnescapedNewline =
          Text.count("\\\n") != FormatTok.NewlinesBefore;
      FormatTok.WhiteSpaceLength += FormatTok.Tok.getLength();

      if (FormatTok.Tok.is(tok::eof))
        return FormatTok;
      Lex.LexFromRawLexer(FormatTok.Tok);
      Text = rawTokenText(FormatTok.Tok);
    }

    // Now FormatTok is the next non-whitespace token.
    FormatTok.TokenLength = Text.size();

    // In case the token starts with escaped newlines, we want to
    // take them into account as whitespace - this pattern is quite frequent
    // in macro definitions.
    // FIXME: What do we want to do with other escaped spaces, and escaped
    // spaces or newlines in the middle of tokens?
    // FIXME: Add a more explicit test.
    unsigned i = 0;
    while (i + 1 < Text.size() && Text[i] == '\\' && Text[i + 1] == '\n') {
      FormatTok.WhiteSpaceLength += 2;
      FormatTok.TokenLength -= 2;
      i += 2;
    }

    if (FormatTok.Tok.is(tok::raw_identifier)) {
      IdentifierInfo &Info = IdentTable.get(Text);
      FormatTok.Tok.setIdentifierInfo(&Info);
      FormatTok.Tok.setKind(Info.getTokenID());
    }

    if (FormatTok.Tok.is(tok::greatergreater)) {
      FormatTok.Tok.setKind(tok::greater);
      GreaterStashed = true;
    }

    return FormatTok;
  }

private:
  FormatToken FormatTok;
  bool GreaterStashed;
  Lexer &Lex;
  SourceManager &SourceMgr;
  IdentifierTable IdentTable;

  /// Returns the text of \c FormatTok.
  StringRef rawTokenText(Token &Tok) {
    return StringRef(SourceMgr.getCharacterData(Tok.getLocation()),
                     Tok.getLength());
  }
};

class Formatter : public UnwrappedLineConsumer {
public:
  Formatter(const FormatStyle &Style, Lexer &Lex, SourceManager &SourceMgr,
            const std::vector<CharSourceRange> &Ranges)
      : Style(Style), Lex(Lex), SourceMgr(SourceMgr), Ranges(Ranges),
        StructuralError(false) {
  }

  virtual ~Formatter() {
  }

  tooling::Replacements format() {
    LexerBasedFormatTokenSource Tokens(Lex, SourceMgr);
    UnwrappedLineParser Parser(Style, Tokens, *this);
    StructuralError = Parser.parse();
    unsigned PreviousEndOfLineColumn = 0;
    for (std::vector<UnwrappedLine>::iterator I = UnwrappedLines.begin(),
                                              E = UnwrappedLines.end();
         I != E; ++I)
      PreviousEndOfLineColumn =
          formatUnwrappedLine(*I, PreviousEndOfLineColumn);
    return Replaces;
  }

private:
  virtual void consumeUnwrappedLine(const UnwrappedLine &TheLine) {
    UnwrappedLines.push_back(TheLine);
  }

  unsigned formatUnwrappedLine(const UnwrappedLine &TheLine,
                               unsigned PreviousEndOfLineColumn) {
    if (TheLine.Tokens.empty())
      return 0;  // FIXME: Find out how this can ever happen.

    CharSourceRange LineRange =
        CharSourceRange::getTokenRange(TheLine.Tokens.front().Tok.getLocation(),
                                       TheLine.Tokens.back().Tok.getLocation());

    for (unsigned i = 0, e = Ranges.size(); i != e; ++i) {
      if (SourceMgr.isBeforeInTranslationUnit(LineRange.getEnd(),
                                              Ranges[i].getBegin()) ||
          SourceMgr.isBeforeInTranslationUnit(Ranges[i].getEnd(),
                                              LineRange.getBegin()))
        continue;

      TokenAnnotator Annotator(TheLine, Style, SourceMgr, Lex);
      if (!Annotator.annotate())
        break;
      UnwrappedLineFormatter Formatter(
          Style, SourceMgr, TheLine, PreviousEndOfLineColumn,
          Annotator.getLineType(), Annotator.getAnnotations(), Replaces,
          StructuralError);
      return Formatter.format();
    }
    // If we did not reformat this unwrapped line, the column at the end of the
    // last token is unchanged - thus, we can calculate the end of the last
    // token, and return the result.
    const FormatToken &Token = TheLine.Tokens.back();
    return SourceMgr.getSpellingColumnNumber(Token.Tok.getLocation()) +
           Lex.MeasureTokenLength(Token.Tok.getLocation(), SourceMgr,
                                  Lex.getLangOpts()) -
           1;
  }

  FormatStyle Style;
  Lexer &Lex;
  SourceManager &SourceMgr;
  tooling::Replacements Replaces;
  std::vector<CharSourceRange> Ranges;
  std::vector<UnwrappedLine> UnwrappedLines;
  bool StructuralError;
};

tooling::Replacements reformat(const FormatStyle &Style, Lexer &Lex,
                               SourceManager &SourceMgr,
                               std::vector<CharSourceRange> Ranges) {
  Formatter formatter(Style, Lex, SourceMgr, Ranges);
  return formatter.format();
}

}  // namespace format
}  // namespace clang
