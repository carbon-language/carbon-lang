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

#include "BreakableToken.h"
#include "TokenAnnotator.h"
#include "UnwrappedLineParser.h"
#include "WhitespaceManager.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/OperatorPrecedence.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include <queue>
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
  LLVMStyle.ConstructorInitializerAllOnOneLineOrOnePerLine = false;
  LLVMStyle.AllowShortIfStatementsOnASingleLine = false;
  LLVMStyle.ObjCSpaceBeforeProtocolList = true;
  LLVMStyle.PenaltyExcessCharacter = 1000000;
  LLVMStyle.PenaltyReturnTypeOnItsOwnLine = 75;
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
  GoogleStyle.BinPackParameters = true;
  GoogleStyle.AllowAllParametersOfDeclarationOnNextLine = true;
  GoogleStyle.ConstructorInitializerAllOnOneLineOrOnePerLine = true;
  GoogleStyle.AllowShortIfStatementsOnASingleLine = false;
  GoogleStyle.ObjCSpaceBeforeProtocolList = false;
  GoogleStyle.PenaltyExcessCharacter = 1000000;
  GoogleStyle.PenaltyReturnTypeOnItsOwnLine = 200;
  return GoogleStyle;
}

FormatStyle getChromiumStyle() {
  FormatStyle ChromiumStyle = getGoogleStyle();
  ChromiumStyle.AllowAllParametersOfDeclarationOnNextLine = false;
  ChromiumStyle.BinPackParameters = false;
  ChromiumStyle.Standard = FormatStyle::LS_Cpp03;
  ChromiumStyle.DerivePointerBinding = false;
  return ChromiumStyle;
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

class UnwrappedLineFormatter {
public:
  UnwrappedLineFormatter(const FormatStyle &Style, SourceManager &SourceMgr,
                         const AnnotatedLine &Line, unsigned FirstIndent,
                         const AnnotatedToken &RootToken,
                         WhitespaceManager &Whitespaces)
      : Style(Style), SourceMgr(SourceMgr), Line(Line),
        FirstIndent(FirstIndent), RootToken(RootToken),
        Whitespaces(Whitespaces), Count(0) {}

  /// \brief Formats an \c UnwrappedLine.
  ///
  /// \returns The column after the last token in the last line of the
  /// \c UnwrappedLine.
  unsigned format(const AnnotatedLine *NextLine) {
    // Initialize state dependent on indent.
    LineState State;
    State.Column = FirstIndent;
    State.NextToken = &RootToken;
    State.Stack.push_back(
        ParenState(FirstIndent, FirstIndent, !Style.BinPackParameters,
                   /*HasMultiParameterLine=*/ false));
    State.LineContainsContinuedForLoopSection = false;
    State.ParenLevel = 0;
    State.StartOfStringLiteral = 0;
    State.StartOfLineLevel = State.ParenLevel;

    // The first token has already been indented and thus consumed.
    moveStateToNextToken(State, /*DryRun=*/ false);

    // If everything fits on a single line, just put it there.
    unsigned ColumnLimit = Style.ColumnLimit;
    if (NextLine && NextLine->InPPDirective &&
        !NextLine->First.FormatTok.HasUnescapedNewline)
      ColumnLimit = getColumnLimit();
    if (Line.Last->TotalLength <= ColumnLimit - FirstIndent) {
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
          HasMultiParameterLine(HasMultiParameterLine), ColonPos(0),
          StartOfFunctionCall(0), NestedNameSpecifierContinuation(0),
          CallContinuation(0), VariablePos(0) {}

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

    /// \brief The start of the most recent function in a builder-type call.
    unsigned StartOfFunctionCall;

    /// \brief If a nested name specifier was broken over multiple lines, this
    /// contains the start column of the second line. Otherwise 0.
    unsigned NestedNameSpecifierContinuation;

    /// \brief If a call expression was broken over multiple lines, this
    /// contains the start column of the second line. Otherwise 0.
    unsigned CallContinuation;

    /// \brief The column of the first variable name in a variable declaration.
    ///
    /// Used to align further variables if necessary.
    unsigned VariablePos;

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
      if (StartOfFunctionCall != Other.StartOfFunctionCall)
        return StartOfFunctionCall < Other.StartOfFunctionCall;
      if (NestedNameSpecifierContinuation !=
          Other.NestedNameSpecifierContinuation)
        return NestedNameSpecifierContinuation <
               Other.NestedNameSpecifierContinuation;
      if (CallContinuation != Other.CallContinuation)
        return CallContinuation < Other.CallContinuation;
      if (VariablePos != Other.VariablePos)
        return VariablePos < Other.VariablePos;
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

    /// \brief \c true if this line contains a continued for-loop section.
    bool LineContainsContinuedForLoopSection;

    /// \brief The level of nesting inside (), [], <> and {}.
    unsigned ParenLevel;

    /// \brief The \c ParenLevel at the start of this line.
    unsigned StartOfLineLevel;

    /// \brief The start column of the string literal, if we're in a string
    /// literal sequence, 0 otherwise.
    unsigned StartOfStringLiteral;

    /// \brief A stack keeping track of properties applying to parenthesis
    /// levels.
    std::vector<ParenState> Stack;

    /// \brief Comparison operator to be able to used \c LineState in \c map.
    bool operator<(const LineState &Other) const {
      if (NextToken != Other.NextToken)
        return NextToken < Other.NextToken;
      if (Column != Other.Column)
        return Column < Other.Column;
      if (LineContainsContinuedForLoopSection !=
          Other.LineContainsContinuedForLoopSection)
        return LineContainsContinuedForLoopSection;
      if (ParenLevel != Other.ParenLevel)
        return ParenLevel < Other.ParenLevel;
      if (StartOfLineLevel != Other.StartOfLineLevel)
        return StartOfLineLevel < Other.StartOfLineLevel;
      if (StartOfStringLiteral != Other.StartOfStringLiteral)
        return StartOfStringLiteral < Other.StartOfStringLiteral;
      return Stack < Other.Stack;
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
  unsigned addTokenToState(bool Newline, bool DryRun, LineState &State) {
    const AnnotatedToken &Current = *State.NextToken;
    const AnnotatedToken &Previous = *State.NextToken->Parent;

    if (State.Stack.size() == 0 || Current.Type == TT_ImplicitStringLiteral) {
      State.Column += State.NextToken->FormatTok.WhiteSpaceLength +
                      State.NextToken->FormatTok.TokenLength;
      if (State.NextToken->Children.empty())
        State.NextToken = NULL;
      else
        State.NextToken = &State.NextToken->Children[0];
      return 0;
    }

    // If we are continuing an expression, we want to indent an extra 4 spaces.
    unsigned ContinuationIndent =
        std::max(State.Stack.back().LastSpace, State.Stack.back().Indent) + 4;
    if (Newline) {
      unsigned WhitespaceStartColumn = State.Column;
      if (Current.is(tok::r_brace)) {
        State.Column = Line.Level * 2;
      } else if (Current.is(tok::string_literal) &&
                 State.StartOfStringLiteral != 0) {
        State.Column = State.StartOfStringLiteral;
        State.Stack.back().BreakBeforeParameter = true;
      } else if (Current.is(tok::lessless) &&
                 State.Stack.back().FirstLessLess != 0) {
        State.Column = State.Stack.back().FirstLessLess;
      } else if (Previous.is(tok::coloncolon)) {
        if (State.Stack.back().NestedNameSpecifierContinuation == 0) {
          State.Column = ContinuationIndent;
          State.Stack.back().NestedNameSpecifierContinuation = State.Column;
        } else {
          State.Column = State.Stack.back().NestedNameSpecifierContinuation;
        }
      } else if (Current.isOneOf(tok::period, tok::arrow)) {
        if (State.Stack.back().CallContinuation == 0) {
          State.Column = ContinuationIndent;
          State.Stack.back().CallContinuation = State.Column;
        } else {
          State.Column = State.Stack.back().CallContinuation;
        }
      } else if (Current.Type == TT_ConditionalExpr) {
        State.Column = State.Stack.back().QuestionColumn;
      } else if (Previous.is(tok::comma) &&
                 State.Stack.back().VariablePos != 0) {
        State.Column = State.Stack.back().VariablePos;
      } else if (Previous.ClosesTemplateDeclaration ||
                 (Current.Type == TT_StartOfName && State.ParenLevel == 0)) {
        State.Column = State.Stack.back().Indent;
      } else if (Current.Type == TT_ObjCSelectorName) {
        if (State.Stack.back().ColonPos > Current.FormatTok.TokenLength) {
          State.Column =
              State.Stack.back().ColonPos - Current.FormatTok.TokenLength;
        } else {
          State.Column = State.Stack.back().Indent;
          State.Stack.back().ColonPos =
              State.Column + Current.FormatTok.TokenLength;
        }
      } else if (Current.Type == TT_StartOfName || Previous.is(tok::equal) ||
                 Previous.Type == TT_ObjCMethodExpr) {
        State.Column = ContinuationIndent;
      } else {
        State.Column = State.Stack.back().Indent;
        // Ensure that we fall back to indenting 4 spaces instead of just
        // flushing continuations left.
        if (State.Column == FirstIndent)
          State.Column += 4;
      }

      if (Current.is(tok::question))
        State.Stack.back().BreakBeforeParameter = true;
      if (Previous.isOneOf(tok::comma, tok::semi) &&
          !State.Stack.back().AvoidBinPacking)
        State.Stack.back().BreakBeforeParameter = false;

      if (!DryRun) {
        unsigned NewLines = 1;
        if (Current.Type == TT_LineComment)
          NewLines =
              std::max(NewLines, std::min(Current.FormatTok.NewlinesBefore,
                                          Style.MaxEmptyLinesToKeep + 1));
        if (!Line.InPPDirective)
          Whitespaces.replaceWhitespace(Current, NewLines, State.Column,
                                        WhitespaceStartColumn);
        else
          Whitespaces.replacePPWhitespace(Current, NewLines, State.Column,
                                          WhitespaceStartColumn);
      }

      State.Stack.back().LastSpace = State.Column;
      State.StartOfLineLevel = State.ParenLevel;

      // Any break on this level means that the parent level has been broken
      // and we need to avoid bin packing there.
      for (unsigned i = 0, e = State.Stack.size() - 1; i != e; ++i) {
        State.Stack[i].BreakBeforeParameter = true;
      }
      const AnnotatedToken *TokenBefore = Current.getPreviousNoneComment();
      if (TokenBefore && !TokenBefore->isOneOf(tok::comma, tok::semi) &&
          !TokenBefore->opensScope())
        State.Stack.back().BreakBeforeParameter = true;

      // If we break after {, we should also break before the corresponding }.
      if (Previous.is(tok::l_brace))
        State.Stack.back().BreakBeforeClosingBrace = true;

      if (State.Stack.back().AvoidBinPacking) {
        // If we are breaking after '(', '{', '<', this is not bin packing
        // unless AllowAllParametersOfDeclarationOnNextLine is false.
        if ((Previous.isNot(tok::l_paren) && Previous.isNot(tok::l_brace)) ||
            (!Style.AllowAllParametersOfDeclarationOnNextLine &&
             Line.MustBeDeclaration))
          State.Stack.back().BreakBeforeParameter = true;
      }
    } else {
      if (Current.is(tok::equal) &&
          (RootToken.is(tok::kw_for) || State.ParenLevel == 0) &&
          State.Stack.back().VariablePos == 0) {
        State.Stack.back().VariablePos = State.Column;
        // Move over * and & if they are bound to the variable name.
        const AnnotatedToken *Tok = &Previous;
        while (Tok &&
               State.Stack.back().VariablePos >= Tok->FormatTok.TokenLength) {
          State.Stack.back().VariablePos -= Tok->FormatTok.TokenLength;
          if (Tok->SpacesRequiredBefore != 0)
            break;
          Tok = Tok->Parent;
        }
        if (Previous.PartOfMultiVariableDeclStmt)
          State.Stack.back().LastSpace = State.Stack.back().VariablePos;
      }

      unsigned Spaces = State.NextToken->SpacesRequiredBefore;

      if (!DryRun)
        Whitespaces.replaceWhitespace(Current, 0, Spaces, State.Column);

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

      if (Previous.opensScope() && Previous.Type != TT_ObjCMethodExpr &&
          Current.Type != TT_LineComment)
        State.Stack.back().Indent = State.Column + Spaces;
      if (Previous.is(tok::comma) && !Current.isTrailingComment())
        State.Stack.back().HasMultiParameterLine = true;

      State.Column += Spaces;
      if (Current.is(tok::l_paren) && Previous.isOneOf(tok::kw_if, tok::kw_for))
        // Treat the condition inside an if as if it was a second function
        // parameter, i.e. let nested calls have an indent of 4.
        State.Stack.back().LastSpace = State.Column + 1; // 1 is length of "(".
      else if (Previous.is(tok::comma))
        State.Stack.back().LastSpace = State.Column;
      else if ((Previous.Type == TT_BinaryOperator ||
                Previous.Type == TT_ConditionalExpr ||
                Previous.Type == TT_CtorInitializerColon) &&
               getPrecedence(Previous) != prec::Assignment)
        State.Stack.back().LastSpace = State.Column;
      else if (Previous.Type == TT_InheritanceColon)
        State.Stack.back().Indent = State.Column;
      else if (Previous.opensScope() && Previous.ParameterCount > 1)
        // If this function has multiple parameters, indent nested calls from
        // the start of the first parameter.
        State.Stack.back().LastSpace = State.Column;
    }

    return moveStateToNextToken(State, DryRun);
  }

  /// \brief Mark the next token as consumed in \p State and modify its stacks
  /// accordingly.
  unsigned moveStateToNextToken(LineState &State, bool DryRun) {
    const AnnotatedToken &Current = *State.NextToken;
    assert(State.Stack.size());

    if (Current.Type == TT_InheritanceColon)
      State.Stack.back().AvoidBinPacking = true;
    if (Current.is(tok::lessless) && State.Stack.back().FirstLessLess == 0)
      State.Stack.back().FirstLessLess = State.Column;
    if (Current.is(tok::question))
      State.Stack.back().QuestionColumn = State.Column;
    if (Current.isOneOf(tok::period, tok::arrow) &&
        Line.Type == LT_BuilderTypeCall && State.ParenLevel == 0)
      State.Stack.back().StartOfFunctionCall =
          Current.LastInChainOfCalls ? 0 : State.Column;
    if (Current.Type == TT_CtorInitializerColon) {
      State.Stack.back().Indent = State.Column + 2;
      if (Style.ConstructorInitializerAllOnOneLineOrOnePerLine)
        State.Stack.back().AvoidBinPacking = true;
      State.Stack.back().BreakBeforeParameter = false;
    }

    // If return returns a binary expression, align after it.
    if (Current.is(tok::kw_return) && !Current.FakeLParens.empty())
      State.Stack.back().LastSpace = State.Column + 7;

    // In ObjC method declaration we align on the ":" of parameters, but we need
    // to ensure that we indent parameters on subsequent lines by at least 4.
    if (Current.Type == TT_ObjCMethodSpecifier)
      State.Stack.back().Indent += 4;

    // Insert scopes created by fake parenthesis.
    const AnnotatedToken *Previous = Current.getPreviousNoneComment();
    // Don't add extra indentation for the first fake parenthesis after
    // 'return', assignements or opening <({[. The indentation for these cases
    // is special cased.
    bool SkipFirstExtraIndent =
        Current.is(tok::kw_return) ||
        (Previous && (Previous->opensScope() ||
                      getPrecedence(*Previous) == prec::Assignment));
    for (SmallVector<prec::Level, 4>::const_reverse_iterator
             I = Current.FakeLParens.rbegin(),
             E = Current.FakeLParens.rend();
         I != E; ++I) {
      ParenState NewParenState = State.Stack.back();
      NewParenState.Indent =
          std::max(std::max(State.Column, NewParenState.Indent),
                   State.Stack.back().LastSpace);

      // Always indent conditional expressions. Never indent expression where
      // the 'operator' is ',', ';' or an assignment (i.e. *I <=
      // prec::Assignment) as those have different indentation rules. Indent
      // other expression, unless the indentation needs to be skipped.
      if (*I == prec::Conditional ||
          (!SkipFirstExtraIndent && *I > prec::Assignment))
        NewParenState.Indent += 4;
      if (Previous && !Previous->opensScope())
        NewParenState.BreakBeforeParameter = false;
      State.Stack.push_back(NewParenState);
      SkipFirstExtraIndent = false;
    }

    // If we encounter an opening (, [, { or <, we add a level to our stacks to
    // prepare for the following tokens.
    if (Current.opensScope()) {
      unsigned NewIndent;
      bool AvoidBinPacking;
      if (Current.is(tok::l_brace)) {
        NewIndent = 2 + State.Stack.back().LastSpace;
        AvoidBinPacking = false;
      } else {
        NewIndent = 4 + std::max(State.Stack.back().LastSpace,
                                 State.Stack.back().StartOfFunctionCall);
        AvoidBinPacking =
            !Style.BinPackParameters || State.Stack.back().AvoidBinPacking;
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
    if (Current.isOneOf(tok::r_paren, tok::r_square) ||
        (Current.is(tok::r_brace) && State.NextToken != &RootToken) ||
        State.NextToken->Type == TT_TemplateCloser) {
      State.Stack.pop_back();
      --State.ParenLevel;
    }

    // Remove scopes created by fake parenthesis.
    for (unsigned i = 0, e = Current.FakeRParens; i != e; ++i) {
      unsigned VariablePos = State.Stack.back().VariablePos;
      State.Stack.pop_back();
      State.Stack.back().VariablePos = VariablePos;
    }

    if (Current.is(tok::string_literal)) {
      State.StartOfStringLiteral = State.Column;
    } else if (Current.isNot(tok::comment)) {
      State.StartOfStringLiteral = 0;
    }

    State.Column += Current.FormatTok.TokenLength;

    if (State.NextToken->Children.empty())
      State.NextToken = NULL;
    else
      State.NextToken = &State.NextToken->Children[0];

    return breakProtrudingToken(Current, State, DryRun);
  }

  /// \brief If the current token sticks out over the end of the line, break
  /// it if possible.
  unsigned breakProtrudingToken(const AnnotatedToken &Current, LineState &State,
                                bool DryRun) {
    llvm::OwningPtr<BreakableToken> Token;
    unsigned StartColumn = State.Column - Current.FormatTok.TokenLength;
    if (Current.is(tok::string_literal)) {
      // Only break up default narrow strings.
      const char *LiteralData = SourceMgr.getCharacterData(
          Current.FormatTok.getStartOfNonWhitespace());
      if (!LiteralData || *LiteralData != '"')
        return 0;

      Token.reset(new BreakableStringLiteral(SourceMgr, Current.FormatTok,
                                             StartColumn));
    } else if (Current.Type == TT_BlockComment) {
      BreakableBlockComment *BBC =
          new BreakableBlockComment(SourceMgr, Current, StartColumn);
      if (!DryRun)
        BBC->alignLines(Whitespaces);
      Token.reset(BBC);
    } else if (Current.Type == TT_LineComment) {
      Token.reset(new BreakableLineComment(SourceMgr, Current, StartColumn));
    } else {
      return 0;
    }

    bool BreakInserted = false;
    unsigned Penalty = 0;
    for (unsigned LineIndex = 0; LineIndex < Token->getLineCount();
         ++LineIndex) {
      unsigned TailOffset = 0;
      unsigned RemainingLength =
          Token->getLineLengthAfterSplit(LineIndex, TailOffset);
      while (RemainingLength > getColumnLimit()) {
        BreakableToken::Split Split =
            Token->getSplit(LineIndex, TailOffset, getColumnLimit());
        if (Split.first == StringRef::npos)
          break;
        assert(Split.first != 0);
        unsigned NewRemainingLength = Token->getLineLengthAfterSplit(
            LineIndex, TailOffset + Split.first + Split.second);
        if (NewRemainingLength >= RemainingLength)
          break;
        if (!DryRun) {
          Token->insertBreak(LineIndex, TailOffset, Split, Line.InPPDirective,
                             Whitespaces);
        }
        TailOffset += Split.first + Split.second;
        RemainingLength = NewRemainingLength;
        Penalty += Style.PenaltyExcessCharacter;
        BreakInserted = true;
      }
      State.Column = RemainingLength;
      if (!DryRun) {
        Token->trimLine(LineIndex, TailOffset, Line.InPPDirective, Whitespaces);
      }
    }

    if (BreakInserted) {
      for (unsigned i = 0, e = State.Stack.size(); i != e; ++i)
        State.Stack[i].BreakBeforeParameter = true;
      State.Stack.back().LastSpace = StartColumn;
    }
    return Penalty;
  }

  unsigned getColumnLimit() {
    // In preprocessor directives reserve two chars for trailing " \"
    return Style.ColumnLimit - (Line.InPPDirective ? 2 : 0);
  }

  /// \brief An edge in the solution space from \c Previous->State to \c State,
  /// inserting a newline dependent on the \c NewLine.
  struct StateNode {
    StateNode(const LineState &State, bool NewLine, StateNode *Previous)
        : State(State), NewLine(NewLine), Previous(Previous) {}
    LineState State;
    bool NewLine;
    StateNode *Previous;
  };

  /// \brief A pair of <penalty, count> that is used to prioritize the BFS on.
  ///
  /// In case of equal penalties, we want to prefer states that were inserted
  /// first. During state generation we make sure that we insert states first
  /// that break the line as late as possible.
  typedef std::pair<unsigned, unsigned> OrderedPenalty;

  /// \brief An item in the prioritized BFS search queue. The \c StateNode's
  /// \c State has the given \c OrderedPenalty.
  typedef std::pair<OrderedPenalty, StateNode *> QueueItem;

  /// \brief The BFS queue type.
  typedef std::priority_queue<QueueItem, std::vector<QueueItem>,
                              std::greater<QueueItem> > QueueType;

  /// \brief Analyze the entire solution space starting from \p InitialState.
  ///
  /// This implements a variant of Dijkstra's algorithm on the graph that spans
  /// the solution space (\c LineStates are the nodes). The algorithm tries to
  /// find the shortest path (the one with lowest penalty) from \p InitialState
  /// to a state where all tokens are placed.
  unsigned analyzeSolutionSpace(LineState &InitialState) {
    std::set<LineState> Seen;

    // Insert start element into queue.
    StateNode *Node =
        new (Allocator.Allocate()) StateNode(InitialState, false, NULL);
    Queue.push(QueueItem(OrderedPenalty(0, Count), Node));
    ++Count;

    // While not empty, take first element and follow edges.
    while (!Queue.empty()) {
      unsigned Penalty = Queue.top().first.first;
      StateNode *Node = Queue.top().second;
      if (Node->State.NextToken == NULL) {
        DEBUG(llvm::errs() << "\n---\nPenalty for line: " << Penalty << "\n");
        break;
      }
      Queue.pop();

      if (!Seen.insert(Node->State).second)
        // State already examined with lower penalty.
        continue;

      addNextStateToQueue(Penalty, Node, /*NewLine=*/ false);
      addNextStateToQueue(Penalty, Node, /*NewLine=*/ true);
    }

    if (Queue.empty())
      // We were unable to find a solution, do nothing.
      // FIXME: Add diagnostic?
      return 0;

    // Reconstruct the solution.
    reconstructPath(InitialState, Queue.top().second);
    DEBUG(llvm::errs() << "---\n");

    // Return the column after the last token of the solution.
    return Queue.top().second->State.Column;
  }

  void reconstructPath(LineState &State, StateNode *Current) {
    // FIXME: This recursive implementation limits the possible number
    // of tokens per line if compiled into a binary with small stack space.
    // To become more independent of stack frame limitations we would need
    // to also change the TokenAnnotator.
    if (Current->Previous == NULL)
      return;
    reconstructPath(State, Current->Previous);
    DEBUG({
      if (Current->NewLine) {
        llvm::errs()
            << "Penalty for splitting before "
            << Current->Previous->State.NextToken->FormatTok.Tok.getName()
            << ": " << Current->Previous->State.NextToken->SplitPenalty << "\n";
      }
    });
    addTokenToState(Current->NewLine, false, State);
  }

  /// \brief Add the following state to the analysis queue \c Queue.
  ///
  /// Assume the current state is \p PreviousNode and has been reached with a
  /// penalty of \p Penalty. Insert a line break if \p NewLine is \c true.
  void addNextStateToQueue(unsigned Penalty, StateNode *PreviousNode,
                           bool NewLine) {
    if (NewLine && !canBreak(PreviousNode->State))
      return;
    if (!NewLine && mustBreak(PreviousNode->State))
      return;
    if (NewLine)
      Penalty += PreviousNode->State.NextToken->SplitPenalty;

    StateNode *Node = new (Allocator.Allocate())
        StateNode(PreviousNode->State, NewLine, PreviousNode);
    Penalty += addTokenToState(NewLine, true, Node->State);
    if (Node->State.Column > getColumnLimit()) {
      unsigned ExcessCharacters = Node->State.Column - getColumnLimit();
      Penalty += Style.PenaltyExcessCharacter * ExcessCharacters;
    }

    Queue.push(QueueItem(OrderedPenalty(Penalty, Count), Node));
    ++Count;
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
    if ((State.NextToken->Parent->isOneOf(tok::comma, tok::semi) ||
         State.NextToken->is(tok::question) ||
         State.NextToken->Type == TT_ConditionalExpr) &&
        State.Stack.back().BreakBeforeParameter &&
        !State.NextToken->isTrailingComment() &&
        State.NextToken->isNot(tok::r_paren) &&
        State.NextToken->isNot(tok::r_brace))
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
    if (State.NextToken->Type == TT_InlineASMColon)
      return true;
    // This prevents breaks like:
    //   ...
    //   SomeParameter, OtherParameter).DoSomething(
    //   ...
    // As they hide "DoSomething" and generally bad for readability.
    if (State.NextToken->isOneOf(tok::period, tok::arrow) &&
        getRemainingLength(State) + State.Column > getColumnLimit() &&
        State.ParenLevel < State.StartOfLineLevel)
      return true;
    return false;
  }

  // Returns the total number of columns required for the remaining tokens.
  unsigned getRemainingLength(const LineState &State) {
    if (State.NextToken && State.NextToken->Parent)
      return Line.Last->TotalLength - State.NextToken->Parent->TotalLength;
    return 0;
  }

  FormatStyle Style;
  SourceManager &SourceMgr;
  const AnnotatedLine &Line;
  const unsigned FirstIndent;
  const AnnotatedToken &RootToken;
  WhitespaceManager &Whitespaces;

  llvm::SpecificBumpPtrAllocator<StateNode> Allocator;
  QueueType Queue;
  // Increasing count of \c StateNode items we have created. This is used
  // to create a deterministic order independent of the container.
  unsigned Count;
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
      if (Newlines > 0)
        FormatTok.LastNewlineOffset =
            FormatTok.WhiteSpaceLength + Text.rfind('\n') + 1;
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

    if (FormatTok.Tok.is(tok::comment)) {
      FormatTok.TrailingWhiteSpaceLength = Text.size() - Text.rtrim().size();
      FormatTok.TokenLength -= FormatTok.TrailingWhiteSpaceLength;
    }

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
      FormatTok.TokenLength = 1;
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
        Whitespaces(SourceMgr, Style), Ranges(Ranges) {}

  virtual ~Formatter() {}

  tooling::Replacements format() {
    LexerBasedFormatTokenSource Tokens(Lex, SourceMgr);
    UnwrappedLineParser Parser(Diag, Style, Tokens, *this);
    bool StructuralError = Parser.parse();
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

    // Adapt level to the next line if this is a comment.
    // FIXME: Can/should this be done in the UnwrappedLineParser?
    const AnnotatedLine *NextNoneCommentLine = NULL;
    for (unsigned i = AnnotatedLines.size() - 1; i > 0; --i) {
      if (NextNoneCommentLine && AnnotatedLines[i].First.is(tok::comment) &&
          AnnotatedLines[i].First.Children.empty())
        AnnotatedLines[i].Level = NextNoneCommentLine->Level;
      else
        NextNoneCommentLine =
            AnnotatedLines[i].First.isNot(tok::r_brace) ? &AnnotatedLines[i]
                                                        : NULL;
    }

    std::vector<int> IndentForLevel;
    bool PreviousLineWasTouched = false;
    const AnnotatedToken *PreviousLineLastToken = 0;
    for (std::vector<AnnotatedLine>::iterator I = AnnotatedLines.begin(),
                                              E = AnnotatedLines.end();
         I != E; ++I) {
      const AnnotatedLine &TheLine = *I;
      const FormatToken &FirstTok = TheLine.First.FormatTok;
      int Offset = getIndentOffset(TheLine.First);
      while (IndentForLevel.size() <= TheLine.Level)
        IndentForLevel.push_back(-1);
      IndentForLevel.resize(TheLine.Level + 1);
      bool WasMoved = PreviousLineWasTouched && FirstTok.NewlinesBefore == 0;
      if (TheLine.First.is(tok::eof)) {
        if (PreviousLineWasTouched) {
          unsigned NewLines = std::min(FirstTok.NewlinesBefore, 1u);
          Whitespaces.replaceWhitespace(TheLine.First, NewLines, /*Indent*/ 0,
                                        /*WhitespaceStartColumn*/ 0);
        }
      } else if (TheLine.Type != LT_Invalid &&
                 (WasMoved || touchesLine(TheLine))) {
        unsigned LevelIndent = getIndent(IndentForLevel, TheLine.Level);
        unsigned Indent = LevelIndent;
        if (static_cast<int>(Indent) + Offset >= 0)
          Indent += Offset;
        if (FirstTok.WhiteSpaceStart.isValid() &&
            // Insert a break even if there is a structural error in case where
            // we break apart a line consisting of multiple unwrapped lines.
            (FirstTok.NewlinesBefore == 0 || !StructuralError)) {
          formatFirstToken(TheLine.First, PreviousLineLastToken, Indent,
                           TheLine.InPPDirective, PreviousEndOfLineColumn);
        } else {
          Indent = LevelIndent =
              SourceMgr.getSpellingColumnNumber(FirstTok.Tok.getLocation()) - 1;
        }
        tryFitMultipleLinesInOne(Indent, I, E);
        UnwrappedLineFormatter Formatter(Style, SourceMgr, TheLine, Indent,
                                         TheLine.First, Whitespaces);
        PreviousEndOfLineColumn =
            Formatter.format(I + 1 != E ? &*(I + 1) : NULL);
        IndentForLevel[TheLine.Level] = LevelIndent;
        PreviousLineWasTouched = true;
      } else {
        if (FirstTok.NewlinesBefore > 0 || FirstTok.IsFirst) {
          unsigned Indent =
              SourceMgr.getSpellingColumnNumber(FirstTok.Tok.getLocation()) - 1;
          unsigned LevelIndent = Indent;
          if (static_cast<int>(LevelIndent) - Offset >= 0)
            LevelIndent -= Offset;
          if (TheLine.First.isNot(tok::comment))
            IndentForLevel[TheLine.Level] = LevelIndent;

          // Remove trailing whitespace of the previous line if it was touched.
          if (PreviousLineWasTouched || touchesEmptyLineBefore(TheLine))
            formatFirstToken(TheLine.First, PreviousLineLastToken, Indent,
                             TheLine.InPPDirective, PreviousEndOfLineColumn);
        }
        // If we did not reformat this unwrapped line, the column at the end of
        // the last token is unchanged - thus, we can calculate the end of the
        // last token.
        SourceLocation LastLoc = TheLine.Last->FormatTok.Tok.getLocation();
        PreviousEndOfLineColumn =
            SourceMgr.getSpellingColumnNumber(LastLoc) +
            Lex.MeasureTokenLength(LastLoc, SourceMgr, Lex.getLangOpts()) - 1;
        PreviousLineWasTouched = false;
        if (TheLine.Last->is(tok::comment))
          Whitespaces.addUntouchableComment(SourceMgr.getSpellingColumnNumber(
              TheLine.Last->FormatTok.Tok.getLocation()) - 1);
      }
      PreviousLineLastToken = I->Last;
    }
    return Whitespaces.generateReplacements();
  }

private:
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

  /// \brief Get the indent of \p Level from \p IndentForLevel.
  ///
  /// \p IndentForLevel must contain the indent for the level \c l
  /// at \p IndentForLevel[l], or a value < 0 if the indent for
  /// that level is unknown.
  unsigned getIndent(const std::vector<int> IndentForLevel, unsigned Level) {
    if (IndentForLevel[Level] != -1)
      return IndentForLevel[Level];
    if (Level == 0)
      return 0;
    return getIndent(IndentForLevel, Level - 1) + 2;
  }

  /// \brief Get the offset of the line relatively to the level.
  ///
  /// For example, 'public:' labels in classes are offset by 1 or 2
  /// characters to the left from their level.
  int getIndentOffset(const AnnotatedToken &RootToken) {
    if (RootToken.isAccessSpecifier(false) || RootToken.isObjCAccessSpecifier())
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
    // We can never merge stuff if there are trailing line comments.
    if (I->Last->Type == TT_LineComment)
      return;

    unsigned Limit = Style.ColumnLimit - Indent;
    // If we already exceed the column limit, we set 'Limit' to 0. The different
    // tryMerge..() functions can then decide whether to still do merging.
    Limit = I->Last->TotalLength > Limit ? 0 : Limit - I->Last->TotalLength;

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
    if (Limit == 0)
      return;
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
    if (Limit == 0)
      return;
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
    if (Line.First.isOneOf(tok::kw_if, tok::kw_while, tok::kw_do, tok::r_brace,
                           tok::kw_else, tok::kw_try, tok::kw_catch,
                           tok::kw_for,
                           // This gets rid of all ObjC @ keywords and methods.
                           tok::at, tok::minus, tok::plus))
      return;

    AnnotatedToken *Tok = &(I + 1)->First;
    if (Tok->Children.empty() && Tok->is(tok::r_brace) &&
        !Tok->MustBreakBefore) {
      // We merge empty blocks even if the line exceeds the column limit.
      Tok->SpacesRequiredBefore = 0;
      Tok->CanBreakBefore = true;
      join(Line, *(I + 1));
      I += 1;
    } else if (Limit != 0) {
      // Check that we still have three lines and they fit into the limit.
      if (I + 2 == E || (I + 2)->Type == LT_Invalid ||
          !nextTwoLinesFitInto(I, Limit))
        return;

      // Second, check that the next line does not contain any braces - if it
      // does, readability declines when putting it into a single line.
      if ((I + 1)->Last->Type == TT_LineComment || Tok->MustBreakBefore)
        return;
      do {
        if (Tok->isOneOf(tok::l_brace, tok::r_brace))
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
    unsigned LengthA = A.Last->TotalLength + B.First.SpacesRequiredBefore;
    A.Last->Children.push_back(B.First);
    while (!A.Last->Children.empty()) {
      A.Last->Children[0].Parent = A.Last;
      A.Last->Children[0].TotalLength += LengthA;
      A.Last = &A.Last->Children[0];
    }
  }

  bool touchesRanges(const CharSourceRange &Range) {
    for (unsigned i = 0, e = Ranges.size(); i != e; ++i) {
      if (!SourceMgr.isBeforeInTranslationUnit(Range.getEnd(),
                                               Ranges[i].getBegin()) &&
          !SourceMgr.isBeforeInTranslationUnit(Ranges[i].getEnd(),
                                               Range.getBegin()))
        return true;
    }
    return false;
  }

  bool touchesLine(const AnnotatedLine &TheLine) {
    const FormatToken *First = &TheLine.First.FormatTok;
    const FormatToken *Last = &TheLine.Last->FormatTok;
    CharSourceRange LineRange = CharSourceRange::getTokenRange(
        First->WhiteSpaceStart.getLocWithOffset(First->LastNewlineOffset),
        Last->Tok.getLocation());
    return touchesRanges(LineRange);
  }

  bool touchesEmptyLineBefore(const AnnotatedLine &TheLine) {
    const FormatToken *First = &TheLine.First.FormatTok;
    CharSourceRange LineRange = CharSourceRange::getCharRange(
        First->WhiteSpaceStart,
        First->WhiteSpaceStart.getLocWithOffset(First->LastNewlineOffset));
    return touchesRanges(LineRange);
  }

  virtual void consumeUnwrappedLine(const UnwrappedLine &TheLine) {
    AnnotatedLines.push_back(AnnotatedLine(TheLine));
  }

  /// \brief Add a new line and the required indent before the first Token
  /// of the \c UnwrappedLine if there was no structural parsing error.
  /// Returns the indent level of the \c UnwrappedLine.
  void formatFirstToken(const AnnotatedToken &RootToken,
                        const AnnotatedToken *PreviousToken, unsigned Indent,
                        bool InPPDirective, unsigned PreviousEndOfLineColumn) {
    const FormatToken &Tok = RootToken.FormatTok;

    unsigned Newlines =
        std::min(Tok.NewlinesBefore, Style.MaxEmptyLinesToKeep + 1);
    if (Newlines == 0 && !Tok.IsFirst)
      Newlines = 1;

    if (!InPPDirective || Tok.HasUnescapedNewline) {
      // Insert extra new line before access specifiers.
      if (PreviousToken && PreviousToken->isOneOf(tok::semi, tok::r_brace) &&
          RootToken.isAccessSpecifier() && Tok.NewlinesBefore == 1)
        ++Newlines;

      Whitespaces.replaceWhitespace(RootToken, Newlines, Indent, 0);
    } else {
      Whitespaces.replacePPWhitespace(RootToken, Newlines, Indent,
                                      PreviousEndOfLineColumn);
    }
  }

  DiagnosticsEngine &Diag;
  FormatStyle Style;
  Lexer &Lex;
  SourceManager &SourceMgr;
  WhitespaceManager Whitespaces;
  std::vector<CharSourceRange> Ranges;
  std::vector<AnnotatedLine> AnnotatedLines;
};

tooling::Replacements reformat(const FormatStyle &Style, Lexer &Lex,
                               SourceManager &SourceMgr,
                               std::vector<CharSourceRange> Ranges,
                               DiagnosticConsumer *DiagClient) {
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
  LangOpts.LineComment = 1;
  LangOpts.Bool = 1;
  LangOpts.ObjC1 = 1;
  LangOpts.ObjC2 = 1;
  return LangOpts;
}

} // namespace format
} // namespace clang
