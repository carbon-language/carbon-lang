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

class AnnotatedToken {
public:
  AnnotatedToken(const FormatToken &FormatTok)
      : FormatTok(FormatTok), Type(TT_Unknown),
        ClosesTemplateDeclaration(false), Parent(NULL) {
  }

  bool is(tok::TokenKind Kind) const {
    return FormatTok.Tok.is(Kind);
  }
  bool isNot(tok::TokenKind Kind) const {
    return FormatTok.Tok.isNot(Kind);
  }
  bool isObjCAtKeyword(tok::ObjCKeywordKind Kind) const {
    return FormatTok.Tok.isObjCAtKeyword(Kind);
  }

  FormatToken FormatTok;

  TokenType Type;

  bool SpaceRequiredBefore;
  bool CanBreakBefore;
  bool MustBreakBefore;

  bool ClosesTemplateDeclaration;

  std::vector<AnnotatedToken> Children;
  AnnotatedToken *Parent;
};

static prec::Level getPrecedence(const AnnotatedToken &Tok) {
  return getBinOpPrecedence(Tok.FormatTok.Tok.getKind(), true, true);
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
  UnwrappedLineFormatter(
      const FormatStyle &Style, SourceManager &SourceMgr,
      const UnwrappedLine &Line, unsigned PreviousEndOfLineColumn,
      LineType CurrentLineType, const AnnotatedToken &RootToken,
      tooling::Replacements &Replaces, bool StructuralError)
      : Style(Style), SourceMgr(SourceMgr), Line(Line),
        PreviousEndOfLineColumn(PreviousEndOfLineColumn),
        CurrentLineType(CurrentLineType), RootToken(RootToken),
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
    State.NextToken = &RootToken;
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
    const AnnotatedToken *Tok = State.NextToken;
    while (Tok != NULL) {
      Columns += (Tok->SpaceRequiredBefore ? 1 : 0) +
                 Tok->FormatTok.TokenLength;
      // A special case for the colon of a constructor initializer as this only
      // needs to be put on a new line if the line needs to be split.
      if (Columns > Style.ColumnLimit - (Line.InPPDirective ? 1 : 0) ||
          (Tok->MustBreakBefore && Tok->Type != TT_CtorInitializerColon)) {
        FitsOnALine = false;
        break;
      }
      Tok = Tok->Children.empty() ? NULL : &Tok->Children[0];
    }

    // Start iterating at 1 as we have correctly formatted of Token #0 above.
    while (State.NextToken != NULL) {
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

    const AnnotatedToken *NextToken;

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
      if (Other.NextToken != NextToken)
        return Other.NextToken > NextToken;
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
    const FormatToken &Current = State.NextToken->FormatTok;
    const FormatToken &Previous = State.NextToken->Parent->FormatTok;
    unsigned ParenLevel = State.Indent.size() - 1;

    if (Newline) {
      unsigned WhitespaceStartColumn = State.Column;
      if (Previous.Tok.is(tok::l_brace)) {
        // FIXME: This does not work with nested static initializers.
        // Implement a better handling for static initializers and similar
        // constructs.
        State.Column = Line.Level * 2 + 2;
      } else if (Current.Tok.is(tok::string_literal) &&
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
      } else if (RootToken.is(tok::kw_for) && Previous.Tok.is(tok::comma)) {
        State.Column = State.ForLoopVariablePos;
      } else if (State.NextToken->Parent->ClosesTemplateDeclaration) {
        State.Column = State.Indent[ParenLevel] - 4;
      } else {
        State.Column = State.Indent[ParenLevel];
      }

      State.StartOfLineLevel = ParenLevel + 1;

      if (RootToken.is(tok::kw_for))
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
          State.NextToken->Type != TT_ConditionalExpr)
        State.Indent[ParenLevel] += 2;
    } else {
      if (Current.Tok.is(tok::equal) && RootToken.is(tok::kw_for))
        State.ForLoopVariablePos = State.Column - Previous.TokenLength;

      unsigned Spaces = State.NextToken->SpaceRequiredBefore ? 1 : 0;
      if (State.NextToken->Type == TT_LineComment)
        Spaces = Style.SpacesBeforeTrailingComments;

      if (!DryRun)
        replaceWhitespace(Current, 0, Spaces);

      if (RootToken.isNot(tok::kw_for) &&
          (getPrecedence(Previous) == prec::Assignment ||
           Previous.Tok.is(tok::kw_return)))
        State.Indent[ParenLevel] = State.Column + Spaces;
      if (Previous.Tok.is(tok::l_paren) ||
          State.NextToken->Parent->Type == TT_TemplateOpener)
        State.Indent[ParenLevel] = State.Column;

      // Top-level spaces that are not part of assignments are exempt as that
      // mostly leads to better results.
      State.Column += Spaces;
      if (Spaces > 0 &&
          (ParenLevel != 0 || getPrecedence(Previous) == prec::Assignment))
        State.LastSpace[ParenLevel] = State.Column;
    }
    moveStateToNextToken(State);
  }

  /// \brief Mark the next token as consumed in \p State and modify its stacks
  /// accordingly.
  void moveStateToNextToken(IndentState &State) {
    const AnnotatedToken &Current = *State.NextToken;
    unsigned ParenLevel = State.Indent.size() - 1;

    if (Current.is(tok::lessless) && State.FirstLessLess[ParenLevel] == 0)
      State.FirstLessLess[ParenLevel] = State.Column;

    State.Column += Current.FormatTok.TokenLength;

    // If we encounter an opening (, [, { or <, we add a level to our stacks to
    // prepare for the following tokens.
    if (Current.is(tok::l_paren) || Current.is(tok::l_square) ||
        Current.is(tok::l_brace) ||
        State.NextToken->Type == TT_TemplateOpener) {
      State.Indent.push_back(4 + State.LastSpace.back());
      State.LastSpace.push_back(State.LastSpace.back());
      State.FirstLessLess.push_back(0);
    }

    // If we encounter a closing ), ], } or >, we can remove a level from our
    // stacks.
    if (Current.is(tok::r_paren) || Current.is(tok::r_square) ||
        (Current.is(tok::r_brace) && State.NextToken != &RootToken) ||
        State.NextToken->Type == TT_TemplateCloser) {
      State.Indent.pop_back();
      State.LastSpace.pop_back();
      State.FirstLessLess.pop_back();
    }
    if (State.NextToken->Children.empty())
      State.NextToken = NULL;
    else
      State.NextToken = &State.NextToken->Children[0];
  }

  /// \brief Calculate the penalty for splitting after the token at \p Index.
  unsigned splitPenalty(const AnnotatedToken &Tok) {
    const AnnotatedToken &Left = Tok;
    const AnnotatedToken &Right = Tok.Children[0];

    // In for-loops, prefer breaking at ',' and ';'.
    if (RootToken.is(tok::kw_for) &&
        (Left.isNot(tok::comma) && Left.isNot(tok::semi)))
      return 20;

    if (Left.is(tok::semi) || Left.is(tok::comma) ||
        Left.ClosesTemplateDeclaration)
      return 0;
    if (Left.is(tok::l_paren))
      return 20;

    prec::Level Level = getPrecedence(Left);

    // Breaking after an assignment leads to a bad result as the two sides of
    // the assignment are visually very close together.
    if (Level == prec::Assignment)
      return 50;

    if (Level != prec::Unknown)
      return Level;

    if (Right.is(tok::arrow) || Right.is(tok::period))
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
    if (State.NextToken == NULL)
      return 0;

    if (!NewLine && State.NextToken->MustBreakBefore)
      return UINT_MAX;
    if (NewLine && !State.NextToken->CanBreakBefore)
      return UINT_MAX;
    if (!NewLine && State.NextToken->Parent->is(tok::semi) &&
        State.LineContainsContinuedForLoopSection)
      return UINT_MAX;

    unsigned CurrentPenalty = 0;
    if (NewLine) {
      CurrentPenalty += Parameters.PenaltyIndentLevel * State.Indent.size() +
                        splitPenalty(*State.NextToken->Parent);
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
      unsigned Offset = std::min<int>(Style.ColumnLimit - 1,
                                      WhitespaceStartColumn);
      for (unsigned i = 0; i < NewLines; ++i) {
        NewLineText += std::string(Style.ColumnLimit - Offset - 1, ' ');
        NewLineText += "\\\n";
        Offset = 0;
      }
    }
    Replaces.insert(tooling::Replacement(SourceMgr, Tok.WhiteSpaceStart,
                                         Tok.WhiteSpaceLength, NewLineText +
                                         std::string(Spaces, ' ')));
  }

  /// \brief Add a new line and the required indent before the first Token
  /// of the \c UnwrappedLine if there was no structural parsing error.
  /// Returns the indent level of the \c UnwrappedLine.
  unsigned formatFirstToken() {
    const FormatToken &Tok = RootToken.FormatTok;
    if (!Tok.WhiteSpaceStart.isValid() || StructuralError)
      return SourceMgr.getSpellingColumnNumber(Tok.Tok.getLocation()) - 1;

    unsigned Newlines = std::min(Tok.NewlinesBefore,
                                 Style.MaxEmptyLinesToKeep + 1);
    if (Newlines == 0 && !Tok.IsFirst)
      Newlines = 1;
    unsigned Indent = Line.Level * 2;

    bool IsAccessModifier = false;
    if (RootToken.is(tok::kw_public) || RootToken.is(tok::kw_protected) ||
        RootToken.is(tok::kw_private))
      IsAccessModifier = true;
    else if (RootToken.is(tok::at) && !RootToken.Children.empty() &&
             (RootToken.Children[0].isObjCAtKeyword(tok::objc_public) ||
              RootToken.Children[0].isObjCAtKeyword(tok::objc_protected) ||
              RootToken.Children[0].isObjCAtKeyword(tok::objc_package) ||
              RootToken.Children[0].isObjCAtKeyword(tok::objc_private)))
      IsAccessModifier = true;

    if (IsAccessModifier &&
        static_cast<int>(Indent) + Style.AccessModifierOffset >= 0)
      Indent += Style.AccessModifierOffset;
    if (!Line.InPPDirective || Tok.HasUnescapedNewline)
      replaceWhitespace(Tok, Newlines, Indent);
    else
      replacePPWhitespace(Tok, Newlines, Indent, PreviousEndOfLineColumn);
    return Indent;
  }

  FormatStyle Style;
  SourceManager &SourceMgr;
  const UnwrappedLine &Line;
  const unsigned PreviousEndOfLineColumn;
  const LineType CurrentLineType;
  const AnnotatedToken &RootToken;
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
      : Style(Style), SourceMgr(SourceMgr), Lex(Lex),
        RootToken(Line.RootToken) {
  }

  /// \brief A parser that gathers additional information about tokens.
  ///
  /// The \c TokenAnnotator tries to matches parenthesis and square brakets and
  /// store a parenthesis levels. It also tries to resolve matching "<" and ">"
  /// into template parameter lists.
  class AnnotatingParser {
  public:
    AnnotatingParser(AnnotatedToken &RootToken)
        : CurrentToken(&RootToken), KeywordVirtualFound(false) {
    }

    bool parseAngle() {
      while (CurrentToken != NULL) {
        if (CurrentToken->is(tok::greater)) {
          CurrentToken->Type = TT_TemplateCloser;
          next();
          return true;
        }
        if (CurrentToken->is(tok::r_paren) || CurrentToken->is(tok::r_square) ||
            CurrentToken->is(tok::r_brace))
          return false;
        if (CurrentToken->is(tok::pipepipe) || CurrentToken->is(tok::ampamp) ||
            CurrentToken->is(tok::question) || CurrentToken->is(tok::colon))
          return false;
        if (!consumeToken())
          return false;
      }
      return false;
    }

    bool parseParens() {
      while (CurrentToken != NULL) {
        if (CurrentToken->is(tok::r_paren)) {
          next();
          return true;
        }
        if (CurrentToken->is(tok::r_square) || CurrentToken->is(tok::r_brace))
          return false;
        if (!consumeToken())
          return false;
      }
      return false;
    }

    bool parseSquare() {
      while (CurrentToken != NULL) {
        if (CurrentToken->is(tok::r_square)) {
          next();
          return true;
        }
        if (CurrentToken->is(tok::r_paren) || CurrentToken->is(tok::r_brace))
          return false;
        if (!consumeToken())
          return false;
      }
      return false;
    }

    bool parseConditional() {
      while (CurrentToken != NULL) {
        if (CurrentToken->is(tok::colon)) {
          CurrentToken->Type = TT_ConditionalExpr;
          next();
          return true;
        }
        if (!consumeToken())
          return false;
      }
      return false;
    }

    bool parseTemplateDeclaration() {
      if (CurrentToken != NULL && CurrentToken->is(tok::less)) {
        CurrentToken->Type = TT_TemplateOpener;
        next();
        if (!parseAngle())
          return false;
        CurrentToken->Parent->ClosesTemplateDeclaration = true;
        parseLine();
        return true;
      }
      return false;
    }

    bool consumeToken() {
      AnnotatedToken *Tok = CurrentToken;
      next();
      switch (Tok->FormatTok.Tok.getKind()) {
      case tok::l_paren:
        if (!parseParens())
          return false;
        if (CurrentToken != NULL && CurrentToken->is(tok::colon)) {
          CurrentToken->Type = TT_CtorInitializerColon;
          next();
        }
        break;
      case tok::l_square:
        if (!parseSquare())
          return false;
        break;
      case tok::less:
        if (parseAngle())
          Tok->Type = TT_TemplateOpener;
        else {
          Tok->Type = TT_BinaryOperator;
          CurrentToken = Tok;
          next();
        }
        break;
      case tok::r_paren:
      case tok::r_square:
        return false;
      case tok::greater:
        Tok->Type = TT_BinaryOperator;
        break;
      case tok::kw_operator:
        if (CurrentToken->is(tok::l_paren)) {
          CurrentToken->Type = TT_OverloadedOperator;
          next();
          if (CurrentToken != NULL && CurrentToken->is(tok::r_paren)) {
            CurrentToken->Type = TT_OverloadedOperator;
            next();
          }
        } else {
          while (CurrentToken != NULL && CurrentToken->isNot(tok::l_paren)) {
            CurrentToken->Type = TT_OverloadedOperator;
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
      while (CurrentToken != NULL) {
        if (CurrentToken->is(tok::slash))
          CurrentToken->Type = TT_DirectorySeparator;
        else if (CurrentToken->is(tok::less))
          CurrentToken->Type = TT_TemplateOpener;
        else if (CurrentToken->is(tok::greater))
          CurrentToken->Type = TT_TemplateCloser;
        next();
      }
    }

    void parsePreprocessorDirective() {
      next();
      if (CurrentToken == NULL)
        return;
      // Hashes in the middle of a line can lead to any strange token
      // sequence.
      if (CurrentToken->FormatTok.Tok.getIdentifierInfo() == NULL)
        return;
      switch (
          CurrentToken->FormatTok.Tok.getIdentifierInfo()->getPPKeywordID()) {
      case tok::pp_include:
      case tok::pp_import:
        parseIncludeDirective();
        break;
      default:
        break;
      }
    }

    LineType parseLine() {
      if (CurrentToken->is(tok::hash)) {
        parsePreprocessorDirective();
        return LT_PreprocessorDirective;
      }
      while (CurrentToken != NULL) {
        if (CurrentToken->is(tok::kw_virtual))
          KeywordVirtualFound = true;
        if (!consumeToken())
          return LT_Invalid;
      }
      if (KeywordVirtualFound)
        return LT_VirtualFunctionDecl;
      return LT_Other;
    }

    void next() {
      if (CurrentToken != NULL && !CurrentToken->Children.empty())
        CurrentToken = &CurrentToken->Children[0];
      else
        CurrentToken = NULL;
    }

  private:
    AnnotatedToken *CurrentToken;
    bool KeywordVirtualFound;
  };

  void createAnnotatedTokens(AnnotatedToken &Current) {
    if (!Current.FormatTok.Children.empty()) {
      Current.Children.push_back(AnnotatedToken(Current.FormatTok.Children[0]));
      Current.Children.back().Parent = &Current;
      createAnnotatedTokens(Current.Children.back());
    }
  }

  void calculateExtraInformation(AnnotatedToken &Current) {
    Current.SpaceRequiredBefore = spaceRequiredBefore(Current);

    if (Current.Type == TT_CtorInitializerColon || Current.Parent->Type ==
        TT_LineComment || (Current.is(tok::string_literal) &&
                           Current.Parent->is(tok::string_literal))) {
      Current.MustBreakBefore = true;
    } else if (Current.is(tok::at) && Current.Parent->Parent->is(tok::at)) {
      // Don't put two objc's '@' on the same line. This could happen,
      // as in, @optional @property ...
      Current.MustBreakBefore = true;
    } else {
      Current.MustBreakBefore = false;
    }

    Current.CanBreakBefore = Current.MustBreakBefore || canBreakBefore(Current);

    if (!Current.Children.empty())
      calculateExtraInformation(Current.Children[0]);
  }

  bool annotate() {
    createAnnotatedTokens(RootToken);

    AnnotatingParser Parser(RootToken);
    CurrentLineType = Parser.parseLine();
    if (CurrentLineType == LT_Invalid)
      return false;

    determineTokenTypes(RootToken, /*IsRHS=*/false);

    if (RootToken.Type == TT_ObjCMethodSpecifier)
      CurrentLineType = LT_ObjCMethodDecl;

    if (!RootToken.Children.empty())
      calculateExtraInformation(RootToken.Children[0]);
    return true;
  }

  LineType getLineType() {
    return CurrentLineType;
  }

  const AnnotatedToken &getRootToken() {
    return RootToken;
  }

private:
  void determineTokenTypes(AnnotatedToken &Current, bool IsRHS) {
    if (getPrecedence(Current) == prec::Assignment ||
        Current.is(tok::kw_return) || Current.is(tok::kw_throw))
      IsRHS = true;

    if (Current.Type == TT_Unknown) {
      if (Current.is(tok::star) || Current.is(tok::amp)) {
        Current.Type = determineStarAmpUsage(Current, IsRHS);
      } else if (Current.is(tok::minus) || Current.is(tok::plus)) {
        Current.Type = determinePlusMinusUsage(Current);
      } else if (Current.is(tok::minusminus) || Current.is(tok::plusplus)) {
        Current.Type = determineIncrementUsage(Current);
      } else if (Current.is(tok::exclaim)) {
        Current.Type = TT_UnaryOperator;
      } else if (isBinaryOperator(Current)) {
        Current.Type = TT_BinaryOperator;
      } else if (Current.is(tok::comment)) {
        std::string Data(Lexer::getSpelling(Current.FormatTok.Tok, SourceMgr,
                                            Lex.getLangOpts()));
        if (StringRef(Data).startswith("//"))
          Current.Type = TT_LineComment;
        else
          Current.Type = TT_BlockComment;
      }
    }

    if (!Current.Children.empty())
      determineTokenTypes(Current.Children[0], IsRHS);
  }

  bool isBinaryOperator(const AnnotatedToken &Tok) {
    // Comma is a binary operator, but does not behave as such wrt. formatting.
    return getPrecedence(Tok) > prec::Comma;
  }

  TokenType determineStarAmpUsage(const AnnotatedToken &Tok, bool IsRHS) {
    if (Tok.Parent == NULL)
      return TT_UnaryOperator;
    if (Tok.Children.size() == 0)
      return TT_Unknown;
    const FormatToken &PrevToken = Tok.Parent->FormatTok;
    const FormatToken &NextToken = Tok.Children[0].FormatTok;

    if (PrevToken.Tok.is(tok::l_paren) || PrevToken.Tok.is(tok::l_square) ||
        PrevToken.Tok.is(tok::comma) || PrevToken.Tok.is(tok::kw_return) ||
        PrevToken.Tok.is(tok::colon) || Tok.Parent->Type == TT_BinaryOperator)
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

  TokenType determinePlusMinusUsage(const AnnotatedToken &Tok) {
    // At the start of the line, +/- specific ObjectiveC method declarations.
    if (Tok.Parent == NULL)
      return TT_ObjCMethodSpecifier;

    // Use heuristics to recognize unary operators.
    if (Tok.Parent->is(tok::equal) || Tok.Parent->is(tok::l_paren) ||
        Tok.Parent->is(tok::comma) || Tok.Parent->is(tok::l_square) ||
        Tok.Parent->is(tok::question) || Tok.Parent->is(tok::colon) ||
        Tok.Parent->is(tok::kw_return) || Tok.Parent->is(tok::kw_case))
      return TT_UnaryOperator;

    // There can't be to consecutive binary operators.
    if (Tok.Parent->Type == TT_BinaryOperator)
      return TT_UnaryOperator;

    // Fall back to marking the token as binary operator.
    return TT_BinaryOperator;
  }

  /// \brief Determine whether ++/-- are pre- or post-increments/-decrements.
  TokenType determineIncrementUsage(const AnnotatedToken &Tok) {
    if (Tok.Parent != NULL && Tok.Parent->is(tok::identifier))
      return TT_TrailingUnaryOperator;

    return TT_UnaryOperator;
  }

  bool spaceRequiredBetween(const AnnotatedToken &Left,
                            const AnnotatedToken &Right) {
    if (Right.is(tok::hashhash))
      return Left.is(tok::hash);
    if (Left.is(tok::hashhash) || Left.is(tok::hash))
      return Right.is(tok::hash);
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
      return Left.FormatTok.Tok.isLiteral() ||
             (Left.isNot(tok::star) && Left.isNot(tok::amp) &&
              !Style.PointerAndReferenceBindToType);
    if (Left.is(tok::amp) || Left.is(tok::star))
      return Right.FormatTok.Tok.isLiteral() ||
             Style.PointerAndReferenceBindToType;
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
    if (Right.is(tok::l_paren)) {
      return Left.is(tok::kw_if) || Left.is(tok::kw_for) ||
             Left.is(tok::kw_while) || Left.is(tok::kw_switch) ||
             (Left.isNot(tok::identifier) && Left.isNot(tok::kw_sizeof) &&
              Left.isNot(tok::kw_typeof) && Left.isNot(tok::kw_alignof));
    }
    if (Left.is(tok::at) &&
        Right.FormatTok.Tok.getObjCKeywordID() != tok::objc_not_keyword)
      return false;
    return true;
  }

  bool spaceRequiredBefore(const AnnotatedToken &Tok) {
    if (CurrentLineType == LT_ObjCMethodDecl) {
      if (Tok.is(tok::identifier) && !Tok.Children.empty() &&
          Tok.Children[0].is(tok::colon) && Tok.Parent->is(tok::identifier))
        return true;
      if (Tok.is(tok::colon))
        return false;
      if (Tok.Parent->Type == TT_ObjCMethodSpecifier)
        return true;
      if (Tok.Parent->is(tok::r_paren) && Tok.is(tok::identifier))
        // Don't space between ')' and <id>
        return false;
      if (Tok.Parent->is(tok::colon) && Tok.is(tok::l_paren))
        // Don't space between ':' and '('
        return false;
    }

    if (Tok.Type == TT_CtorInitializerColon)
      return true;
    if (Tok.Type == TT_OverloadedOperator)
      return Tok.is(tok::identifier) || Tok.is(tok::kw_new) ||
             Tok.is(tok::kw_delete);
    if (Tok.Parent->Type == TT_OverloadedOperator)
      return false;
    if (Tok.is(tok::colon))
      return RootToken.isNot(tok::kw_case) && (!Tok.Children.empty());
    if (Tok.Parent->Type == TT_UnaryOperator)
      return false;
    if (Tok.Type == TT_UnaryOperator)
      return Tok.Parent->isNot(tok::l_paren) &&
             Tok.Parent->isNot(tok::l_square);
    if (Tok.Parent->is(tok::greater) && Tok.is(tok::greater)) {
      return Tok.Type == TT_TemplateCloser && Tok.Parent->Type ==
             TT_TemplateCloser && Style.SplitTemplateClosingGreater;
    }
    if (Tok.Type == TT_DirectorySeparator ||
        Tok.Parent->Type == TT_DirectorySeparator)
      return false;
    if (Tok.Type == TT_BinaryOperator || Tok.Parent->Type == TT_BinaryOperator)
      return true;
    if (Tok.Parent->Type == TT_TemplateCloser && Tok.is(tok::l_paren))
      return false;
    if (Tok.is(tok::less) && RootToken.is(tok::hash))
      return true;
    if (Tok.Type == TT_TrailingUnaryOperator)
      return false;
    return spaceRequiredBetween(*Tok.Parent, Tok);
  }

  bool canBreakBefore(const AnnotatedToken &Right) {
    const AnnotatedToken &Left = *Right.Parent;
    if (CurrentLineType == LT_ObjCMethodDecl) {
      if (Right.is(tok::identifier) && !Right.Children.empty() &&
          Right.Children[0].is(tok::colon) && Left.is(tok::identifier))
        return true;
      if (CurrentLineType == LT_ObjCMethodDecl && Right.is(tok::identifier) &&
          Left.is(tok::l_paren) && Left.Parent->is(tok::colon))
        // Don't break this identifier as ':' or identifier
        // before it will break.
        return false;
      if (Right.is(tok::colon) && Left.is(tok::identifier) &&
          Left.CanBreakBefore)
        // Don't break at ':' if identifier before it can beak.
        return false;
    }
    if (Left.ClosesTemplateDeclaration)
      return true;
    if (Left.Type == TT_PointerOrReference || Left.Type == TT_TemplateCloser ||
        Right.Type == TT_ConditionalExpr) {
      return false;
    }
    if (Left.is(tok::equal) && CurrentLineType == LT_VirtualFunctionDecl)
      return false;

    if (Right.is(tok::r_paren) || Right.is(tok::l_brace) ||
        Right.is(tok::comment) || Right.is(tok::greater))
      return false;
    return (isBinaryOperator(Left) && Left.isNot(tok::lessless)) ||
           Left.is(tok::comma) || Right.is(tok::lessless) ||
           Right.is(tok::arrow) || Right.is(tok::period) ||
           Right.is(tok::colon) || Left.is(tok::semi) ||
           Left.is(tok::l_brace) ||
           (Left.is(tok::l_paren) && !Right.is(tok::r_paren));
  }

  FormatStyle Style;
  SourceManager &SourceMgr;
  Lexer &Lex;
  LineType CurrentLineType;
  AnnotatedToken RootToken;
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
      FormatTok.HasUnescapedNewline = Text.count("\\\n") !=
                                      FormatTok.NewlinesBefore;
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
      PreviousEndOfLineColumn = formatUnwrappedLine(*I,
                                                    PreviousEndOfLineColumn);
    return Replaces;
  }

private:
  virtual void consumeUnwrappedLine(const UnwrappedLine &TheLine) {
    UnwrappedLines.push_back(TheLine);
  }

  unsigned formatUnwrappedLine(const UnwrappedLine &TheLine,
                               unsigned PreviousEndOfLineColumn) {
    const FormatToken *First = &TheLine.RootToken;
    const FormatToken *Last = First;
    while (!Last->Children.empty())
      Last = &Last->Children.back();
    CharSourceRange LineRange = CharSourceRange::getTokenRange(
                                    First->Tok.getLocation(),
                                    Last->Tok.getLocation());

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
          Annotator.getLineType(), Annotator.getRootToken(), Replaces,
          StructuralError);
      return Formatter.format();
    }
    // If we did not reformat this unwrapped line, the column at the end of the
    // last token is unchanged - thus, we can calculate the end of the last
    // token, and return the result.
    return SourceMgr.getSpellingColumnNumber(Last->Tok.getLocation()) +
           Lex.MeasureTokenLength(Last->Tok.getLocation(), SourceMgr,
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

} // namespace format
} // namespace clang
