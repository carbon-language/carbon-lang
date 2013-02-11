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
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "format-formatter"

#include "TokenAnnotator.h"
#include "UnwrappedLineParser.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/OperatorPrecedence.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/Debug.h"
#include <string>

namespace clang {
namespace format {

FormatStyle getLLVMStyle() {
  FormatStyle LLVMStyle;
  LLVMStyle.ColumnLimit = 80;
  LLVMStyle.MaxEmptyLinesToKeep = 1;
  LLVMStyle.PointerBindsToType = false;
  LLVMStyle.DerivePointerBinding = false;
  LLVMStyle.AccessModifierOffset = -2;
  LLVMStyle.Standard = FormatStyle::LS_Cpp03;
  LLVMStyle.IndentCaseLabels = false;
  LLVMStyle.SpacesBeforeTrailingComments = 1;
  LLVMStyle.BinPackParameters = true;
  LLVMStyle.AllowAllParametersOfDeclarationOnNextLine = true;
  LLVMStyle.AllowReturnTypeOnItsOwnLine = true;
  LLVMStyle.ConstructorInitializerAllOnOneLineOrOnePerLine = false;
  LLVMStyle.AllowShortIfStatementsOnASingleLine = false;
  LLVMStyle.ObjCSpaceBeforeProtocolList = true;
  LLVMStyle.PenaltyExcessCharacter = 1000000;
  return LLVMStyle;
}

FormatStyle getGoogleStyle() {
  FormatStyle GoogleStyle;
  GoogleStyle.ColumnLimit = 80;
  GoogleStyle.MaxEmptyLinesToKeep = 1;
  GoogleStyle.PointerBindsToType = true;
  GoogleStyle.DerivePointerBinding = true;
  GoogleStyle.AccessModifierOffset = -1;
  GoogleStyle.Standard = FormatStyle::LS_Auto;
  GoogleStyle.IndentCaseLabels = true;
  GoogleStyle.SpacesBeforeTrailingComments = 2;
  GoogleStyle.BinPackParameters = false;
  GoogleStyle.AllowAllParametersOfDeclarationOnNextLine = true;
  GoogleStyle.AllowReturnTypeOnItsOwnLine = false;
  GoogleStyle.ConstructorInitializerAllOnOneLineOrOnePerLine = true;
  GoogleStyle.AllowShortIfStatementsOnASingleLine = false;
  GoogleStyle.ObjCSpaceBeforeProtocolList = false;
  GoogleStyle.PenaltyExcessCharacter = 1000000;
  return GoogleStyle;
}

FormatStyle getChromiumStyle() {
  FormatStyle ChromiumStyle = getGoogleStyle();
  ChromiumStyle.AllowAllParametersOfDeclarationOnNextLine = false;
  ChromiumStyle.Standard = FormatStyle::LS_Cpp03;
  ChromiumStyle.DerivePointerBinding = false;
  return ChromiumStyle;
}

static bool isTrailingComment(const AnnotatedToken &Tok) {
  return Tok.is(tok::comment) &&
         (Tok.Children.empty() || Tok.Children[0].MustBreakBefore);
}

// Returns the length of everything up to the first possible line break after
// the ), ], } or > matching \c Tok.
static unsigned getLengthToMatchingParen(const AnnotatedToken &Tok) {
  if (Tok.MatchingParen == NULL)
    return 0;
  AnnotatedToken *End = Tok.MatchingParen;
  while (!End->Children.empty() && !End->Children[0].CanBreakBefore) {
    End = &End->Children[0];
  }
  return End->TotalLength - Tok.TotalLength + 1;
}

/// \brief Manages the whitespaces around tokens and their replacements.
///
/// This includes special handling for certain constructs, e.g. the alignment of
/// trailing line comments.
class WhitespaceManager {
public:
  WhitespaceManager(SourceManager &SourceMgr) : SourceMgr(SourceMgr) {}

  /// \brief Replaces the whitespace in front of \p Tok. Only call once for
  /// each \c AnnotatedToken.
  void replaceWhitespace(const AnnotatedToken &Tok, unsigned NewLines,
                         unsigned Spaces, unsigned WhitespaceStartColumn,
                         const FormatStyle &Style) {
    // 2+ newlines mean an empty line separating logic scopes.
    if (NewLines >= 2)
      alignComments();

    // Align line comments if they are trailing or if they continue other
    // trailing comments.
    if (isTrailingComment(Tok) && (Tok.Parent != NULL || !Comments.empty())) {
      if (Style.ColumnLimit >=
          Spaces + WhitespaceStartColumn + Tok.FormatTok.TokenLength) {
        Comments.push_back(StoredComment());
        Comments.back().Tok = Tok.FormatTok;
        Comments.back().Spaces = Spaces;
        Comments.back().NewLines = NewLines;
        if (NewLines == 0)
          Comments.back().MinColumn = WhitespaceStartColumn + Spaces;
        else
          Comments.back().MinColumn = Spaces;
        Comments.back().MaxColumn =
            Style.ColumnLimit - Spaces - Tok.FormatTok.TokenLength;
        return;
      }
    }

    // If this line does not have a trailing comment, align the stored comments.
    if (Tok.Children.empty() && !isTrailingComment(Tok))
      alignComments();
    storeReplacement(Tok.FormatTok,
                     std::string(NewLines, '\n') + std::string(Spaces, ' '));
  }

  /// \brief Like \c replaceWhitespace, but additionally adds right-aligned
  /// backslashes to escape newlines inside a preprocessor directive.
  ///
  /// This function and \c replaceWhitespace have the same behavior if
  /// \c Newlines == 0.
  void replacePPWhitespace(const AnnotatedToken &Tok, unsigned NewLines,
                           unsigned Spaces, unsigned WhitespaceStartColumn,
                           const FormatStyle &Style) {
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
    storeReplacement(Tok.FormatTok, NewLineText + std::string(Spaces, ' '));
  }

  /// \brief Returns all the \c Replacements created during formatting.
  const tooling::Replacements &generateReplacements() {
    alignComments();
    return Replaces;
  }

private:
  /// \brief Structure to store a comment for later layout and alignment.
  struct StoredComment {
    FormatToken Tok;
    unsigned MinColumn;
    unsigned MaxColumn;
    unsigned NewLines;
    unsigned Spaces;
  };
  SmallVector<StoredComment, 16> Comments;
  typedef SmallVector<StoredComment, 16>::iterator comment_iterator;

  /// \brief Try to align all stashed comments.
  void alignComments() {
    unsigned MinColumn = 0;
    unsigned MaxColumn = UINT_MAX;
    comment_iterator Start = Comments.begin();
    for (comment_iterator I = Comments.begin(), E = Comments.end(); I != E;
         ++I) {
      if (I->MinColumn > MaxColumn || I->MaxColumn < MinColumn) {
        alignComments(Start, I, MinColumn);
        MinColumn = I->MinColumn;
        MaxColumn = I->MaxColumn;
        Start = I;
      } else {
        MinColumn = std::max(MinColumn, I->MinColumn);
        MaxColumn = std::min(MaxColumn, I->MaxColumn);
      }
    }
    alignComments(Start, Comments.end(), MinColumn);
    Comments.clear();
  }

  /// \brief Put all the comments between \p I and \p E into \p Column.
  void alignComments(comment_iterator I, comment_iterator E, unsigned Column) {
    while (I != E) {
      unsigned Spaces = I->Spaces + Column - I->MinColumn;
      storeReplacement(I->Tok, std::string(I->NewLines, '\n') +
                               std::string(Spaces, ' '));
      ++I;
    }
  }

  /// \brief Stores \p Text as the replacement for the whitespace in front of
  /// \p Tok.
  void storeReplacement(const FormatToken &Tok, const std::string Text) {
    // Don't create a replacement, if it does not change anything.
    if (StringRef(SourceMgr.getCharacterData(Tok.WhiteSpaceStart),
                  Tok.WhiteSpaceLength) == Text)
      return;

    Replaces.insert(tooling::Replacement(SourceMgr, Tok.WhiteSpaceStart,
                                         Tok.WhiteSpaceLength, Text));
  }

  SourceManager &SourceMgr;
  tooling::Replacements Replaces;
};

class UnwrappedLineFormatter {
public:
  UnwrappedLineFormatter(const FormatStyle &Style, SourceManager &SourceMgr,
                         const AnnotatedLine &Line, unsigned FirstIndent,
                         const AnnotatedToken &RootToken,
                         WhitespaceManager &Whitespaces, bool StructuralError)
      : Style(Style), SourceMgr(SourceMgr), Line(Line),
        FirstIndent(FirstIndent), RootToken(RootToken),
        Whitespaces(Whitespaces) {
  }

  /// \brief Formats an \c UnwrappedLine.
  ///
  /// \returns The column after the last token in the last line of the
  /// \c UnwrappedLine.
  unsigned format() {
    // Initialize state dependent on indent.
    LineState State;
    State.Column = FirstIndent;
    State.NextToken = &RootToken;
    State.Stack.push_back(
        ParenState(FirstIndent + 4, FirstIndent, !Style.BinPackParameters,
                   /*HasMultiParameterLine=*/ false));
    State.VariablePos = 0;
    State.LineContainsContinuedForLoopSection = false;
    State.ParenLevel = 0;

    DEBUG({
      DebugTokenState(*State.NextToken);
    });

    // The first token has already been indented and thus consumed.
    moveStateToNextToken(State);

    // If everything fits on a single line, just put it there.
    if (Line.Last->TotalLength <= getColumnLimit() - FirstIndent) {
      while (State.NextToken != NULL) {
        addTokenToState(false, false, State);
      }
      return State.Column;
    }

    // If the ObjC method declaration does not fit on a line, we should format
    // it with one arg per line.
    if (Line.Type == LT_ObjCMethodDecl)
      State.Stack.back().BreakBeforeParameter = true;

    // Find best solution in solution space.
    return analyzeSolutionSpace(State);
  }

private:
  void DebugTokenState(const AnnotatedToken &AnnotatedTok) {
    const Token &Tok = AnnotatedTok.FormatTok.Tok;
    llvm::errs() << StringRef(SourceMgr.getCharacterData(Tok.getLocation()),
                              Tok.getLength());
    llvm::errs();
  }

  struct ParenState {
    ParenState(unsigned Indent, unsigned LastSpace, bool AvoidBinPacking,
               bool HasMultiParameterLine)
        : Indent(Indent), LastSpace(LastSpace), FirstLessLess(0),
          BreakBeforeClosingBrace(false), QuestionColumn(0),
          AvoidBinPacking(AvoidBinPacking), BreakBeforeParameter(false),
          HasMultiParameterLine(HasMultiParameterLine), ColonPos(0) {
    }

    /// \brief The position to which a specific parenthesis level needs to be
    /// indented.
    unsigned Indent;

    /// \brief The position of the last space on each level.
    ///
    /// Used e.g. to break like:
    /// functionCall(Parameter, otherCall(
    ///                             OtherParameter));
    unsigned LastSpace;

    /// \brief The position the first "<<" operator encountered on each level.
    ///
    /// Used to align "<<" operators. 0 if no such operator has been encountered
    /// on a level.
    unsigned FirstLessLess;

    /// \brief Whether a newline needs to be inserted before the block's closing
    /// brace.
    ///
    /// We only want to insert a newline before the closing brace if there also
    /// was a newline after the beginning left brace.
    bool BreakBeforeClosingBrace;

    /// \brief The column of a \c ? in a conditional expression;
    unsigned QuestionColumn;

    /// \brief Avoid bin packing, i.e. multiple parameters/elements on multiple
    /// lines, in this context.
    bool AvoidBinPacking;

    /// \brief Break after the next comma (or all the commas in this context if
    /// \c AvoidBinPacking is \c true).
    bool BreakBeforeParameter;

    /// \brief This context already has a line with more than one parameter.
    bool HasMultiParameterLine;

    /// \brief The position of the colon in an ObjC method declaration/call.
    unsigned ColonPos;

    bool operator<(const ParenState &Other) const {
      if (Indent != Other.Indent)
        return Indent < Other.Indent;
      if (LastSpace != Other.LastSpace)
        return LastSpace < Other.LastSpace;
      if (FirstLessLess != Other.FirstLessLess)
        return FirstLessLess < Other.FirstLessLess;
      if (BreakBeforeClosingBrace != Other.BreakBeforeClosingBrace)
        return BreakBeforeClosingBrace;
      if (QuestionColumn != Other.QuestionColumn)
        return QuestionColumn < Other.QuestionColumn;
      if (AvoidBinPacking != Other.AvoidBinPacking)
        return AvoidBinPacking;
      if (BreakBeforeParameter != Other.BreakBeforeParameter)
        return BreakBeforeParameter;
      if (HasMultiParameterLine != Other.HasMultiParameterLine)
        return HasMultiParameterLine;
      if (ColonPos != Other.ColonPos)
        return ColonPos < Other.ColonPos;
      return false;
    }
  };

  /// \brief The current state when indenting a unwrapped line.
  ///
  /// As the indenting tries different combinations this is copied by value.
  struct LineState {
    /// \brief The number of used columns in the current line.
    unsigned Column;

    /// \brief The token that needs to be next formatted.
    const AnnotatedToken *NextToken;

    /// \brief The column of the first variable name in a variable declaration.
    ///
    /// Used to align further variables if necessary.
    unsigned VariablePos;

    /// \brief \c true if this line contains a continued for-loop section.
    bool LineContainsContinuedForLoopSection;

    /// \brief The level of nesting inside (), [], <> and {}.
    unsigned ParenLevel;

    /// \brief A stack keeping track of properties applying to parenthesis
    /// levels.
    std::vector<ParenState> Stack;

    /// \brief Comparison operator to be able to used \c LineState in \c map.
    bool operator<(const LineState &Other) const {
      if (Other.NextToken != NextToken)
        return Other.NextToken > NextToken;
      if (Other.Column != Column)
        return Other.Column > Column;
      if (Other.VariablePos != VariablePos)
        return Other.VariablePos < VariablePos;
      if (Other.LineContainsContinuedForLoopSection !=
          LineContainsContinuedForLoopSection)
        return LineContainsContinuedForLoopSection;
      if (Other.ParenLevel != ParenLevel)
        return Other.ParenLevel < ParenLevel;
      return Other.Stack < Stack;
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
  void addTokenToState(bool Newline, bool DryRun, LineState &State) {
    const AnnotatedToken &Current = *State.NextToken;
    const AnnotatedToken &Previous = *State.NextToken->Parent;
    assert(State.Stack.size());

    if (Current.Type == TT_ImplicitStringLiteral) {
      State.Column += State.NextToken->FormatTok.WhiteSpaceLength +
                      State.NextToken->FormatTok.TokenLength;
      if (State.NextToken->Children.empty())
        State.NextToken = NULL;
      else
        State.NextToken = &State.NextToken->Children[0];
      return;
    }

    if (Newline) {
      unsigned WhitespaceStartColumn = State.Column;
      if (Current.is(tok::r_brace)) {
        State.Column = Line.Level * 2;
      } else if (Current.is(tok::string_literal) &&
                 Previous.is(tok::string_literal)) {
        State.Column = State.Column - Previous.FormatTok.TokenLength;
      } else if (Current.is(tok::lessless) &&
                 State.Stack.back().FirstLessLess != 0) {
        State.Column = State.Stack.back().FirstLessLess;
      } else if (State.ParenLevel != 0 &&
                 (Previous.is(tok::equal) || Previous.is(tok::coloncolon) ||
                  Current.is(tok::period) || Current.is(tok::arrow) ||
                  Current.is(tok::question))) {
        // Indent and extra 4 spaces after if we know the current expression is
        // continued.  Don't do that on the top level, as we already indent 4
        // there.
        State.Column = std::max(State.Stack.back().LastSpace,
                                State.Stack.back().Indent) + 4;
      } else if (Current.Type == TT_ConditionalExpr) {
        State.Column = State.Stack.back().QuestionColumn;
      } else if (Previous.is(tok::comma) && State.VariablePos != 0 &&
                 ((RootToken.is(tok::kw_for) && State.ParenLevel == 1) ||
                  State.ParenLevel == 0)) {
        State.Column = State.VariablePos;
      } else if (State.NextToken->Parent->ClosesTemplateDeclaration ||
                 Current.Type == TT_StartOfName) {
        State.Column = State.Stack.back().Indent - 4;
      } else if (Current.Type == TT_ObjCSelectorName) {
        if (State.Stack.back().ColonPos > Current.FormatTok.TokenLength) {
          State.Column =
              State.Stack.back().ColonPos - Current.FormatTok.TokenLength;
        } else {
          State.Column = State.Stack.back().Indent;
          State.Stack.back().ColonPos =
              State.Column + Current.FormatTok.TokenLength;
        }
      } else if (Previous.Type == TT_ObjCMethodExpr) {
        State.Column = State.Stack.back().Indent + 4;
      } else {
        State.Column = State.Stack.back().Indent;
      }

      if (Previous.is(tok::comma) && !State.Stack.back().AvoidBinPacking)
        State.Stack.back().BreakBeforeParameter = false;

      if (RootToken.is(tok::kw_for))
        State.LineContainsContinuedForLoopSection = Previous.isNot(tok::semi);

      if (!DryRun) {
        if (!Line.InPPDirective)
          Whitespaces.replaceWhitespace(Current, 1, State.Column,
                                        WhitespaceStartColumn, Style);
        else
          Whitespaces.replacePPWhitespace(Current, 1, State.Column,
                                          WhitespaceStartColumn, Style);
      }

      State.Stack.back().LastSpace = State.Column;
      if (Current.is(tok::colon) && Current.Type != TT_ConditionalExpr)
        State.Stack.back().Indent += 2;
    } else {
      if (Current.is(tok::equal) &&
          (RootToken.is(tok::kw_for) || State.ParenLevel == 0))
        State.VariablePos = State.Column - Previous.FormatTok.TokenLength;

      unsigned Spaces = State.NextToken->SpacesRequiredBefore;

      if (!DryRun)
        Whitespaces.replaceWhitespace(Current, 0, Spaces, State.Column, Style);

      if (Current.Type == TT_ObjCSelectorName &&
          State.Stack.back().ColonPos == 0) {
        if (State.Stack.back().Indent + Current.LongestObjCSelectorName >
            State.Column + Spaces + Current.FormatTok.TokenLength)
          State.Stack.back().ColonPos =
              State.Stack.back().Indent + Current.LongestObjCSelectorName;
        else
          State.Stack.back().ColonPos =
              State.Column + Spaces + Current.FormatTok.TokenLength;
      }

      if (Current.Type != TT_LineComment &&
          (Previous.is(tok::l_paren) || Previous.is(tok::l_brace) ||
           State.NextToken->Parent->Type == TT_TemplateOpener))
        State.Stack.back().Indent = State.Column + Spaces;
      if (Previous.is(tok::comma) && !isTrailingComment(Current))
        State.Stack.back().HasMultiParameterLine = true;

      State.Column += Spaces;
      if (Current.is(tok::l_paren) && Previous.is(tok::kw_if))
        // Treat the condition inside an if as if it was a second function
        // parameter, i.e. let nested calls have an indent of 4.
        State.Stack.back().LastSpace = State.Column + 1; // 1 is length of "(".
      else if (Previous.is(tok::comma) && State.ParenLevel != 0)
        // Top-level spaces are exempt as that mostly leads to better results.
        State.Stack.back().LastSpace = State.Column;
      else if ((Previous.Type == TT_BinaryOperator ||
                Previous.Type == TT_ConditionalExpr ||
                Previous.Type == TT_CtorInitializerColon) &&
               getPrecedence(Previous) != prec::Assignment)
        State.Stack.back().LastSpace = State.Column;
      else if (Previous.ParameterCount > 1 &&
               (Previous.is(tok::l_paren) || Previous.is(tok::l_square) ||
                Previous.is(tok::l_brace) ||
                Previous.Type == TT_TemplateOpener))
        // If this function has multiple parameters, indent nested calls from
        // the start of the first parameter.
        State.Stack.back().LastSpace = State.Column;
    }

    // If we break after an {, we should also break before the corresponding }.
    if (Newline && Previous.is(tok::l_brace))
      State.Stack.back().BreakBeforeClosingBrace = true;

    if (State.Stack.back().AvoidBinPacking && Newline &&
        (Line.First.isNot(tok::kw_for) || State.ParenLevel != 1)) {
      // If we are breaking after '(', '{', '<', this is not bin packing unless
      // AllowAllParametersOfDeclarationOnNextLine is false.
      if ((Previous.isNot(tok::l_paren) && Previous.isNot(tok::l_brace) &&
           Previous.Type != TT_TemplateOpener) ||
          (!Style.AllowAllParametersOfDeclarationOnNextLine &&
           Line.MustBeDeclaration))
        State.Stack.back().BreakBeforeParameter = true;

      // Any break on this level means that the parent level has been broken
      // and we need to avoid bin packing there.
      for (unsigned i = 0, e = State.Stack.size() - 1; i != e; ++i) {
        if (Line.First.isNot(tok::kw_for) || i != 1)
          State.Stack[i].BreakBeforeParameter = true;
      }
    }

    moveStateToNextToken(State);
  }

  /// \brief Mark the next token as consumed in \p State and modify its stacks
  /// accordingly.
  void moveStateToNextToken(LineState &State) {
    const AnnotatedToken &Current = *State.NextToken;
    assert(State.Stack.size());

    if (Current.is(tok::lessless) && State.Stack.back().FirstLessLess == 0)
      State.Stack.back().FirstLessLess = State.Column;
    if (Current.is(tok::question))
      State.Stack.back().QuestionColumn = State.Column;
    if (Current.is(tok::l_brace) && Current.MatchingParen != NULL &&
        !Current.MatchingParen->MustBreakBefore) {
      if (getLengthToMatchingParen(Current) + State.Column > getColumnLimit())
        State.Stack.back().BreakBeforeParameter = true;
    }

    // Insert scopes created by fake parenthesis.
    for (unsigned i = 0, e = Current.FakeLParens; i != e; ++i) {
      ParenState NewParenState = State.Stack.back();
      NewParenState.Indent = std::max(State.Column, State.Stack.back().Indent);
      State.Stack.push_back(NewParenState);
    }

    // If we encounter an opening (, [, { or <, we add a level to our stacks to
    // prepare for the following tokens.
    if (Current.is(tok::l_paren) || Current.is(tok::l_square) ||
        Current.is(tok::l_brace) ||
        State.NextToken->Type == TT_TemplateOpener) {
      unsigned NewIndent;
      bool AvoidBinPacking;
      if (Current.is(tok::l_brace)) {
        NewIndent = 2 + State.Stack.back().LastSpace;
        AvoidBinPacking = false;
      } else {
        NewIndent = 4 + State.Stack.back().LastSpace;
        AvoidBinPacking = !Style.BinPackParameters;
      }
      State.Stack.push_back(
          ParenState(NewIndent, State.Stack.back().LastSpace, AvoidBinPacking,
                     State.Stack.back().HasMultiParameterLine));
      ++State.ParenLevel;
    }

    // If this '[' opens an ObjC call, determine whether all parameters fit into
    // one line and put one per line if they don't.
    if (Current.is(tok::l_square) && Current.Type == TT_ObjCMethodExpr &&
        Current.MatchingParen != NULL) {
      if (getLengthToMatchingParen(Current) + State.Column > getColumnLimit())
        State.Stack.back().BreakBeforeParameter = true;
    }

    // If we encounter a closing ), ], } or >, we can remove a level from our
    // stacks.
    if (Current.is(tok::r_paren) || Current.is(tok::r_square) ||
        (Current.is(tok::r_brace) && State.NextToken != &RootToken) ||
        State.NextToken->Type == TT_TemplateCloser) {
      State.Stack.pop_back();
      --State.ParenLevel;
    }

    // Remove scopes created by fake parenthesis.
    for (unsigned i = 0, e = Current.FakeRParens; i != e; ++i) {
      State.Stack.pop_back();
    }

    if (State.NextToken->Children.empty())
      State.NextToken = NULL;
    else
      State.NextToken = &State.NextToken->Children[0];

    State.Column += Current.FormatTok.TokenLength;
  }

  unsigned getColumnLimit() {
    return Style.ColumnLimit - (Line.InPPDirective ? 1 : 0);
  }

  /// \brief An edge in the solution space starting from the \c LineState and
  /// inserting a newline dependent on the \c bool.
  typedef std::pair<bool, const LineState *> Edge;

  /// \brief An item in the prioritized BFS search queue. The \c LineState was
  /// reached using the \c Edge.
  typedef std::pair<LineState, Edge> QueueItem;

  /// \brief Analyze the entire solution space starting from \p InitialState.
  ///
  /// This implements a variant of Dijkstra's algorithm on the graph that spans
  /// the solution space (\c LineStates are the nodes). The algorithm tries to
  /// find the shortest path (the one with lowest penalty) from \p InitialState
  /// to a state where all tokens are placed.
  unsigned analyzeSolutionSpace(const LineState &InitialState) {
    // Insert start element into queue.
    std::multimap<unsigned, QueueItem> Queue;
    Queue.insert(std::pair<unsigned, QueueItem>(
        0, QueueItem(InitialState, Edge(false, (const LineState *)0))));
    std::map<LineState, Edge> Seen;

    // While not empty, take first element and follow edges.
    while (!Queue.empty()) {
      unsigned Penalty = Queue.begin()->first;
      QueueItem Item = Queue.begin()->second;
      if (Item.first.NextToken == NULL) {
        DEBUG(llvm::errs() << "\n---\nPenalty for line: " << Penalty << "\n");
        break;
      }
      Queue.erase(Queue.begin());

      if (Seen.find(Item.first) != Seen.end())
        continue; // State already examined with lower penalty.

      const LineState &SavedState = Seen.insert(std::pair<LineState, Edge>(
          Item.first,
          Edge(Item.second.first, Item.second.second))).first->first;

      addNextStateToQueue(SavedState, Penalty, /*NewLine=*/ false, Queue);
      addNextStateToQueue(SavedState, Penalty, /*NewLine=*/ true, Queue);
    }

    if (Queue.empty())
      // We were unable to find a solution, do nothing.
      // FIXME: Add diagnostic?
      return 0;

    // Reconstruct the solution.
    // FIXME: Add debugging output.
    Edge *CurrentEdge = &Queue.begin()->second.second;
    while (CurrentEdge->second != NULL) {
      LineState CurrentState = *CurrentEdge->second;
      if (CurrentEdge->first) {
        DEBUG(llvm::errs() << "Penalty for splitting before "
                           << CurrentState.NextToken->FormatTok.Tok.getName()
                           << ": " << CurrentState.NextToken->SplitPenalty
                           << "\n");
      }
      addTokenToState(CurrentEdge->first, false, CurrentState);
      CurrentEdge = &Seen[*CurrentEdge->second];
    }
    DEBUG(llvm::errs() << "---\n");

    // Return the column after the last token of the solution.
    return Queue.begin()->second.first.Column;
  }

  /// \brief Add the following state to the analysis queue \p Queue.
  ///
  /// Assume the current state is \p OldState and has been reached with a
  /// penalty of \p Penalty. Insert a line break if \p NewLine is \c true.
  void addNextStateToQueue(const LineState &OldState, unsigned Penalty,
                           bool NewLine,
                           std::multimap<unsigned, QueueItem> &Queue) {
    if (NewLine && !canBreak(OldState))
      return;
    if (!NewLine && mustBreak(OldState))
      return;
    LineState State(OldState);
    if (NewLine)
      Penalty += State.NextToken->SplitPenalty;
    addTokenToState(NewLine, true, State);
    if (State.Column > getColumnLimit()) {
      unsigned ExcessCharacters = State.Column - getColumnLimit();
      Penalty += Style.PenaltyExcessCharacter * ExcessCharacters;
    }
    Queue.insert(std::pair<unsigned, QueueItem>(
        Penalty, QueueItem(State, Edge(NewLine, &OldState))));
  }

  /// \brief Returns \c true, if a line break after \p State is allowed.
  bool canBreak(const LineState &State) {
    if (!State.NextToken->CanBreakBefore &&
        !(State.NextToken->is(tok::r_brace) &&
          State.Stack.back().BreakBeforeClosingBrace))
      return false;
    // Trying to insert a parameter on a new line if there are already more than
    // one parameter on the current line is bin packing.
    if (State.Stack.back().HasMultiParameterLine &&
        State.Stack.back().AvoidBinPacking)
      return false;
    return true;
  }

  /// \brief Returns \c true, if a line break after \p State is mandatory.
  bool mustBreak(const LineState &State) {
    if (State.NextToken->MustBreakBefore)
      return true;
    if (State.NextToken->is(tok::r_brace) &&
        State.Stack.back().BreakBeforeClosingBrace)
      return true;
    if (State.NextToken->Parent->is(tok::semi) &&
        State.LineContainsContinuedForLoopSection)
      return true;
    if (State.NextToken->Parent->is(tok::comma) &&
        State.Stack.back().BreakBeforeParameter &&
        !isTrailingComment(*State.NextToken))
      return true;
    // FIXME: Comparing LongestObjCSelectorName to 0 is a hacky way of finding
    // out whether it is the first parameter. Clean this up.
    if (State.NextToken->Type == TT_ObjCSelectorName &&
        State.NextToken->LongestObjCSelectorName == 0 &&
        State.Stack.back().BreakBeforeParameter)
      return true;
    if ((State.NextToken->Type == TT_CtorInitializerColon ||
         (State.NextToken->Parent->ClosesTemplateDeclaration &&
          State.ParenLevel == 0)))
      return true;
    return false;
  }

  FormatStyle Style;
  SourceManager &SourceMgr;
  const AnnotatedLine &Line;
  const unsigned FirstIndent;
  const AnnotatedToken &RootToken;
  WhitespaceManager &Whitespaces;
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
      unsigned Newlines = Text.count('\n');
      unsigned EscapedNewlines = Text.count("\\\n");
      FormatTok.NewlinesBefore += Newlines;
      FormatTok.HasUnescapedNewline |= EscapedNewlines != Newlines;
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
      // FIXME: ++FormatTok.NewlinesBefore is missing...
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

  IdentifierTable &getIdentTable() { return IdentTable; }

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
  Formatter(DiagnosticsEngine &Diag, const FormatStyle &Style, Lexer &Lex,
            SourceManager &SourceMgr,
            const std::vector<CharSourceRange> &Ranges)
      : Diag(Diag), Style(Style), Lex(Lex), SourceMgr(SourceMgr),
        Whitespaces(SourceMgr), Ranges(Ranges) {
  }

  virtual ~Formatter() {}

  void deriveLocalStyle() {
    unsigned CountBoundToVariable = 0;
    unsigned CountBoundToType = 0;
    bool HasCpp03IncompatibleFormat = false;
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      if (AnnotatedLines[i].First.Children.empty())
        continue;
      AnnotatedToken *Tok = &AnnotatedLines[i].First.Children[0];
      while (!Tok->Children.empty()) {
        if (Tok->Type == TT_PointerOrReference) {
          bool SpacesBefore = Tok->FormatTok.WhiteSpaceLength > 0;
          bool SpacesAfter = Tok->Children[0].FormatTok.WhiteSpaceLength > 0;
          if (SpacesBefore && !SpacesAfter)
            ++CountBoundToVariable;
          else if (!SpacesBefore && SpacesAfter)
            ++CountBoundToType;
        }

        if (Tok->Type == TT_TemplateCloser &&
            Tok->Parent->Type == TT_TemplateCloser &&
            Tok->FormatTok.WhiteSpaceLength == 0)
          HasCpp03IncompatibleFormat = true;
        Tok = &Tok->Children[0];
      }
    }
    if (Style.DerivePointerBinding) {
      if (CountBoundToType > CountBoundToVariable)
        Style.PointerBindsToType = true;
      else if (CountBoundToType < CountBoundToVariable)
        Style.PointerBindsToType = false;
    }
    if (Style.Standard == FormatStyle::LS_Auto) {
      Style.Standard = HasCpp03IncompatibleFormat ? FormatStyle::LS_Cpp11
                                                  : FormatStyle::LS_Cpp03;
    }
  }

  tooling::Replacements format() {
    LexerBasedFormatTokenSource Tokens(Lex, SourceMgr);
    UnwrappedLineParser Parser(Diag, Style, Tokens, *this);
    StructuralError = Parser.parse();
    unsigned PreviousEndOfLineColumn = 0;
    TokenAnnotator Annotator(Style, SourceMgr, Lex,
                             Tokens.getIdentTable().get("in"));
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      Annotator.annotate(AnnotatedLines[i]);
    }
    deriveLocalStyle();
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      Annotator.calculateFormattingInformation(AnnotatedLines[i]);
    }
    std::vector<int> IndentForLevel;
    for (std::vector<AnnotatedLine>::iterator I = AnnotatedLines.begin(),
                                              E = AnnotatedLines.end();
         I != E; ++I) {
      const AnnotatedLine &TheLine = *I;
      int Offset = GetIndentOffset(TheLine.First);
      while (IndentForLevel.size() <= TheLine.Level)
        IndentForLevel.push_back(-1);
      IndentForLevel.resize(TheLine.Level + 1);
      if (touchesRanges(TheLine) && TheLine.Type != LT_Invalid) {
        unsigned LevelIndent = GetIndent(IndentForLevel, TheLine.Level);
        unsigned Indent = LevelIndent;
        if (static_cast<int>(Indent) + Offset >= 0)
          Indent += Offset;
        if (!TheLine.First.FormatTok.WhiteSpaceStart.isValid() ||
            StructuralError) {
          Indent = LevelIndent = SourceMgr.getSpellingColumnNumber(
              TheLine.First.FormatTok.Tok.getLocation()) - 1;
        } else {
          formatFirstToken(TheLine.First, Indent, TheLine.InPPDirective,
                           PreviousEndOfLineColumn);
        }
        tryFitMultipleLinesInOne(Indent, I, E);
        UnwrappedLineFormatter Formatter(Style, SourceMgr, TheLine, Indent,
                                         TheLine.First, Whitespaces,
                                         StructuralError);
        PreviousEndOfLineColumn = Formatter.format();
        IndentForLevel[TheLine.Level] = LevelIndent;
      } else {
        // If we did not reformat this unwrapped line, the column at the end of
        // the last token is unchanged - thus, we can calculate the end of the
        // last token.
        PreviousEndOfLineColumn =
            SourceMgr.getSpellingColumnNumber(
                TheLine.Last->FormatTok.Tok.getLocation()) +
            Lex.MeasureTokenLength(TheLine.Last->FormatTok.Tok.getLocation(),
                                   SourceMgr, Lex.getLangOpts()) - 1;
        unsigned Indent = SourceMgr.getSpellingColumnNumber(
            TheLine.First.FormatTok.Tok.getLocation()) - 1;
        unsigned LevelIndent = Indent;
        if (static_cast<int>(LevelIndent) - Offset >= 0)
          LevelIndent -= Offset;
        IndentForLevel[TheLine.Level] = LevelIndent;
      }
    }
    return Whitespaces.generateReplacements();
  }

private:
  /// \brief Get the indent of \p Level from \p IndentForLevel.
  ///
  /// \p IndentForLevel must contain the indent for the level \c l
  /// at \p IndentForLevel[l], or a value < 0 if the indent for
  /// that level is unknown.
  unsigned GetIndent(const std::vector<int> IndentForLevel,
                     unsigned Level) {
    if (IndentForLevel[Level] != -1)
      return IndentForLevel[Level];
    if (Level == 0)
      return 0;
    return GetIndent(IndentForLevel, Level - 1) + 2;
  }

  /// \brief Get the offset of the line relatively to the level.
  ///
  /// For example, 'public:' labels in classes are offset by 1 or 2
  /// characters to the left from their level.
  int GetIndentOffset(const AnnotatedToken &RootToken) {
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

    if (IsAccessModifier)
      return Style.AccessModifierOffset;
    return 0;
  }

  /// \brief Tries to merge lines into one.
  ///
  /// This will change \c Line and \c AnnotatedLine to contain the merged line,
  /// if possible; note that \c I will be incremented when lines are merged.
  ///
  /// Returns whether the resulting \c Line can fit in a single line.
  void tryFitMultipleLinesInOne(unsigned Indent,
                                std::vector<AnnotatedLine>::iterator &I,
                                std::vector<AnnotatedLine>::iterator E) {
    unsigned Limit = Style.ColumnLimit - (I->InPPDirective ? 1 : 0) - Indent;

    // We can never merge stuff if there are trailing line comments.
    if (I->Last->Type == TT_LineComment)
      return;

    // Check whether the UnwrappedLine can be put onto a single line. If
    // so, this is bound to be the optimal solution (by definition) and we
    // don't need to analyze the entire solution space.
    if (I->Last->TotalLength > Limit)
      return;
    Limit -= I->Last->TotalLength;

    if (I + 1 == E || (I + 1)->Type == LT_Invalid)
      return;

    if (I->Last->is(tok::l_brace)) {
      tryMergeSimpleBlock(I, E, Limit);
    } else if (I->First.is(tok::kw_if)) {
      tryMergeSimpleIf(I, E, Limit);
    } else if (I->InPPDirective && (I->First.FormatTok.HasUnescapedNewline ||
                                    I->First.FormatTok.IsFirst)) {
      tryMergeSimplePPDirective(I, E, Limit);
    }
    return;
  }

  void tryMergeSimplePPDirective(std::vector<AnnotatedLine>::iterator &I,
                                 std::vector<AnnotatedLine>::iterator E,
                                 unsigned Limit) {
    AnnotatedLine &Line = *I;
    if (!(I + 1)->InPPDirective || (I + 1)->First.FormatTok.HasUnescapedNewline)
      return;
    if (I + 2 != E && (I + 2)->InPPDirective &&
        !(I + 2)->First.FormatTok.HasUnescapedNewline)
      return;
    if (1 + (I + 1)->Last->TotalLength > Limit)
      return;
    join(Line, *(++I));
  }

  void tryMergeSimpleIf(std::vector<AnnotatedLine>::iterator &I,
                        std::vector<AnnotatedLine>::iterator E,
                        unsigned Limit) {
    if (!Style.AllowShortIfStatementsOnASingleLine)
      return;
    if ((I + 1)->InPPDirective != I->InPPDirective ||
        ((I + 1)->InPPDirective &&
         (I + 1)->First.FormatTok.HasUnescapedNewline))
      return;
    AnnotatedLine &Line = *I;
    if (Line.Last->isNot(tok::r_paren))
      return;
    if (1 + (I + 1)->Last->TotalLength > Limit)
      return;
    if ((I + 1)->First.is(tok::kw_if) || (I + 1)->First.Type == TT_LineComment)
      return;
    // Only inline simple if's (no nested if or else).
    if (I + 2 != E && (I + 2)->First.is(tok::kw_else))
      return;
    join(Line, *(++I));
  }

  void tryMergeSimpleBlock(std::vector<AnnotatedLine>::iterator &I,
                           std::vector<AnnotatedLine>::iterator E,
                           unsigned Limit) {
    // First, check that the current line allows merging. This is the case if
    // we're not in a control flow statement and the last token is an opening
    // brace.
    AnnotatedLine &Line = *I;
    bool AllowedTokens =
        Line.First.isNot(tok::kw_if) && Line.First.isNot(tok::kw_while) &&
        Line.First.isNot(tok::kw_do) && Line.First.isNot(tok::r_brace) &&
        Line.First.isNot(tok::kw_else) && Line.First.isNot(tok::kw_try) &&
        Line.First.isNot(tok::kw_catch) && Line.First.isNot(tok::kw_for) &&
        // This gets rid of all ObjC @ keywords and methods.
        Line.First.isNot(tok::at) && Line.First.isNot(tok::minus) &&
        Line.First.isNot(tok::plus);
    if (!AllowedTokens)
      return;

    AnnotatedToken *Tok = &(I + 1)->First;
    if (Tok->Children.empty() && Tok->is(tok::r_brace) &&
        !Tok->MustBreakBefore && Tok->TotalLength <= Limit) {
      Tok->SpacesRequiredBefore = 0;
      join(Line, *(I + 1));
      I += 1;
    } else {
      // Check that we still have three lines and they fit into the limit.
      if (I + 2 == E || (I + 2)->Type == LT_Invalid ||
          !nextTwoLinesFitInto(I, Limit))
        return;

      // Second, check that the next line does not contain any braces - if it
      // does, readability declines when putting it into a single line.
      if ((I + 1)->Last->Type == TT_LineComment || Tok->MustBreakBefore)
        return;
      do {
        if (Tok->is(tok::l_brace) || Tok->is(tok::r_brace))
          return;
        Tok = Tok->Children.empty() ? NULL : &Tok->Children.back();
      } while (Tok != NULL);

      // Last, check that the third line contains a single closing brace.
      Tok = &(I + 2)->First;
      if (!Tok->Children.empty() || Tok->isNot(tok::r_brace) ||
          Tok->MustBreakBefore)
        return;

      join(Line, *(I + 1));
      join(Line, *(I + 2));
      I += 2;
    }
  }

  bool nextTwoLinesFitInto(std::vector<AnnotatedLine>::iterator I,
                           unsigned Limit) {
    return 1 + (I + 1)->Last->TotalLength + 1 + (I + 2)->Last->TotalLength <=
           Limit;
  }

  void join(AnnotatedLine &A, const AnnotatedLine &B) {
    A.Last->Children.push_back(B.First);
    while (!A.Last->Children.empty()) {
      A.Last->Children[0].Parent = A.Last;
      A.Last = &A.Last->Children[0];
    }
  }

  bool touchesRanges(const AnnotatedLine &TheLine) {
    const FormatToken *First = &TheLine.First.FormatTok;
    const FormatToken *Last = &TheLine.Last->FormatTok;
    CharSourceRange LineRange = CharSourceRange::getTokenRange(
        First->Tok.getLocation(), Last->Tok.getLocation());
    for (unsigned i = 0, e = Ranges.size(); i != e; ++i) {
      if (!SourceMgr.isBeforeInTranslationUnit(LineRange.getEnd(),
                                               Ranges[i].getBegin()) &&
          !SourceMgr.isBeforeInTranslationUnit(Ranges[i].getEnd(),
                                               LineRange.getBegin()))
        return true;
    }
    return false;
  }

  virtual void consumeUnwrappedLine(const UnwrappedLine &TheLine) {
    AnnotatedLines.push_back(AnnotatedLine(TheLine));
  }

  /// \brief Add a new line and the required indent before the first Token
  /// of the \c UnwrappedLine if there was no structural parsing error.
  /// Returns the indent level of the \c UnwrappedLine.
  void formatFirstToken(const AnnotatedToken &RootToken, unsigned Indent,
                        bool InPPDirective, unsigned PreviousEndOfLineColumn) {
    const FormatToken &Tok = RootToken.FormatTok;

    unsigned Newlines =
        std::min(Tok.NewlinesBefore, Style.MaxEmptyLinesToKeep + 1);
    if (Newlines == 0 && !Tok.IsFirst)
      Newlines = 1;

    if (!InPPDirective || Tok.HasUnescapedNewline) {
      Whitespaces.replaceWhitespace(RootToken, Newlines, Indent, 0, Style);
    } else {
      Whitespaces.replacePPWhitespace(RootToken, Newlines, Indent,
                                      PreviousEndOfLineColumn, Style);
    }
  }

  DiagnosticsEngine &Diag;
  FormatStyle Style;
  Lexer &Lex;
  SourceManager &SourceMgr;
  WhitespaceManager Whitespaces;
  std::vector<CharSourceRange> Ranges;
  std::vector<AnnotatedLine> AnnotatedLines;
  bool StructuralError;
};

tooling::Replacements
reformat(const FormatStyle &Style, Lexer &Lex, SourceManager &SourceMgr,
         std::vector<CharSourceRange> Ranges, DiagnosticConsumer *DiagClient) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  OwningPtr<DiagnosticConsumer> DiagPrinter;
  if (DiagClient == 0) {
    DiagPrinter.reset(new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts));
    DiagPrinter->BeginSourceFile(Lex.getLangOpts(), Lex.getPP());
    DiagClient = DiagPrinter.get();
  }
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      DiagClient, false);
  Diagnostics.setSourceManager(&SourceMgr);
  Formatter formatter(Diagnostics, Style, Lex, SourceMgr, Ranges);
  return formatter.format();
}

LangOptions getFormattingLangOpts() {
  LangOptions LangOpts;
  LangOpts.CPlusPlus = 1;
  LangOpts.CPlusPlus11 = 1;
  LangOpts.Bool = 1;
  LangOpts.ObjC1 = 1;
  LangOpts.ObjC2 = 1;
  return LangOpts;
}

} // namespace format
} // namespace clang
