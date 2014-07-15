//===--- ContinuationIndenter.cpp - Format C++ code -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the continuation indenter.
///
//===----------------------------------------------------------------------===//

#include "BreakableToken.h"
#include "ContinuationIndenter.h"
#include "WhitespaceManager.h"
#include "clang/Basic/OperatorPrecedence.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Debug.h"
#include <string>

#define DEBUG_TYPE "format-formatter"

namespace clang {
namespace format {

// Returns the length of everything up to the first possible line break after
// the ), ], } or > matching \c Tok.
static unsigned getLengthToMatchingParen(const FormatToken &Tok) {
  if (!Tok.MatchingParen)
    return 0;
  FormatToken *End = Tok.MatchingParen;
  while (End->Next && !End->Next->CanBreakBefore) {
    End = End->Next;
  }
  return End->TotalLength - Tok.TotalLength + 1;
}

// Returns \c true if \c Tok is the "." or "->" of a call and starts the next
// segment of a builder type call.
static bool startsSegmentOfBuilderTypeCall(const FormatToken &Tok) {
  return Tok.isMemberAccess() && Tok.Previous && Tok.Previous->closesScope();
}

// Returns \c true if \c Current starts a new parameter.
static bool startsNextParameter(const FormatToken &Current,
                                const FormatStyle &Style) {
  const FormatToken &Previous = *Current.Previous;
  if (Current.Type == TT_CtorInitializerComma &&
      Style.BreakConstructorInitializersBeforeComma)
    return true;
  return Previous.is(tok::comma) && !Current.isTrailingComment() &&
         (Previous.Type != TT_CtorInitializerComma ||
          !Style.BreakConstructorInitializersBeforeComma);
}

ContinuationIndenter::ContinuationIndenter(const FormatStyle &Style,
                                           SourceManager &SourceMgr,
                                           WhitespaceManager &Whitespaces,
                                           encoding::Encoding Encoding,
                                           bool BinPackInconclusiveFunctions)
    : Style(Style), SourceMgr(SourceMgr), Whitespaces(Whitespaces),
      Encoding(Encoding),
      BinPackInconclusiveFunctions(BinPackInconclusiveFunctions),
      CommentPragmasRegex(Style.CommentPragmas) {}

LineState ContinuationIndenter::getInitialState(unsigned FirstIndent,
                                                const AnnotatedLine *Line,
                                                bool DryRun) {
  LineState State;
  State.FirstIndent = FirstIndent;
  State.Column = FirstIndent;
  State.Line = Line;
  State.NextToken = Line->First;
  State.Stack.push_back(ParenState(FirstIndent, Line->Level, FirstIndent,
                                   /*AvoidBinPacking=*/false,
                                   /*NoLineBreak=*/false));
  State.LineContainsContinuedForLoopSection = false;
  State.StartOfStringLiteral = 0;
  State.StartOfLineLevel = 0;
  State.LowestLevelOnLine = 0;
  State.IgnoreStackForComparison = false;

  // The first token has already been indented and thus consumed.
  moveStateToNextToken(State, DryRun, /*Newline=*/false);
  return State;
}

bool ContinuationIndenter::canBreak(const LineState &State) {
  const FormatToken &Current = *State.NextToken;
  const FormatToken &Previous = *Current.Previous;
  assert(&Previous == Current.Previous);
  if (!Current.CanBreakBefore && !(State.Stack.back().BreakBeforeClosingBrace &&
                                   Current.closesBlockTypeList(Style)))
    return false;
  // The opening "{" of a braced list has to be on the same line as the first
  // element if it is nested in another braced init list or function call.
  if (!Current.MustBreakBefore && Previous.is(tok::l_brace) &&
      Previous.Type != TT_DictLiteral && Previous.BlockKind == BK_BracedInit &&
      Previous.Previous &&
      Previous.Previous->isOneOf(tok::l_brace, tok::l_paren, tok::comma))
    return false;
  // This prevents breaks like:
  //   ...
  //   SomeParameter, OtherParameter).DoSomething(
  //   ...
  // As they hide "DoSomething" and are generally bad for readability.
  if (Previous.opensScope() && Previous.isNot(tok::l_brace) &&
      State.LowestLevelOnLine < State.StartOfLineLevel &&
      State.LowestLevelOnLine < Current.NestingLevel)
    return false;
  if (Current.isMemberAccess() && State.Stack.back().ContainsUnwrappedBuilder)
    return false;

  // Don't create a 'hanging' indent if there are multiple blocks in a single
  // statement.
  if (Style.Language == FormatStyle::LK_JavaScript &&
      Previous.is(tok::l_brace) && State.Stack.size() > 1 &&
      State.Stack[State.Stack.size() - 2].JSFunctionInlined &&
      State.Stack[State.Stack.size() - 2].HasMultipleNestedBlocks)
    return false;

  return !State.Stack.back().NoLineBreak;
}

bool ContinuationIndenter::mustBreak(const LineState &State) {
  const FormatToken &Current = *State.NextToken;
  const FormatToken &Previous = *Current.Previous;
  if (Current.MustBreakBefore || Current.Type == TT_InlineASMColon)
    return true;
  if (State.Stack.back().BreakBeforeClosingBrace &&
      Current.closesBlockTypeList(Style))
    return true;
  if (Previous.is(tok::semi) && State.LineContainsContinuedForLoopSection)
    return true;
  if ((startsNextParameter(Current, Style) || Previous.is(tok::semi) ||
       (Style.BreakBeforeTernaryOperators &&
        (Current.is(tok::question) || (Current.Type == TT_ConditionalExpr &&
                                       Previous.isNot(tok::question)))) ||
       (!Style.BreakBeforeTernaryOperators &&
        (Previous.is(tok::question) || Previous.Type == TT_ConditionalExpr))) &&
      State.Stack.back().BreakBeforeParameter && !Current.isTrailingComment() &&
      !Current.isOneOf(tok::r_paren, tok::r_brace))
    return true;
  if (Style.AlwaysBreakBeforeMultilineStrings &&
      State.Column > State.Stack.back().Indent && // Breaking saves columns.
      !Previous.isOneOf(tok::kw_return, tok::lessless, tok::at) &&
      Previous.Type != TT_InlineASMColon &&
      Previous.Type != TT_ConditionalExpr && nextIsMultilineString(State))
    return true;
  if (((Previous.Type == TT_DictLiteral && Previous.is(tok::l_brace)) ||
       Previous.Type == TT_ArrayInitializerLSquare) &&
      Style.ColumnLimit > 0 &&
      getLengthToMatchingParen(Previous) + State.Column > getColumnLimit(State))
    return true;
  if (Current.Type == TT_CtorInitializerColon &&
      ((Style.AllowShortFunctionsOnASingleLine != FormatStyle::SFS_All) ||
       Style.BreakConstructorInitializersBeforeComma || Style.ColumnLimit != 0))
    return true;

  if (State.Column < getNewLineColumn(State))
    return false;
  if (!Style.BreakBeforeBinaryOperators) {
    // If we need to break somewhere inside the LHS of a binary expression, we
    // should also break after the operator. Otherwise, the formatting would
    // hide the operator precedence, e.g. in:
    //   if (aaaaaaaaaaaaaa ==
    //           bbbbbbbbbbbbbb && c) {..
    // For comparisons, we only apply this rule, if the LHS is a binary
    // expression itself as otherwise, the line breaks seem superfluous.
    // We need special cases for ">>" which we have split into two ">" while
    // lexing in order to make template parsing easier.
    //
    // FIXME: We'll need something similar for styles that break before binary
    // operators.
    bool IsComparison = (Previous.getPrecedence() == prec::Relational ||
                         Previous.getPrecedence() == prec::Equality) &&
                        Previous.Previous &&
                        Previous.Previous->Type != TT_BinaryOperator; // For >>.
    bool LHSIsBinaryExpr =
        Previous.Previous && Previous.Previous->EndsBinaryExpression;
    if (Previous.Type == TT_BinaryOperator &&
        (!IsComparison || LHSIsBinaryExpr) &&
        Current.Type != TT_BinaryOperator && // For >>.
        !Current.isTrailingComment() && !Previous.is(tok::lessless) &&
        Previous.getPrecedence() != prec::Assignment &&
        State.Stack.back().BreakBeforeParameter)
      return true;
  }

  // Same as above, but for the first "<<" operator.
  if (Current.is(tok::lessless) && Current.Type != TT_OverloadedOperator &&
      State.Stack.back().BreakBeforeParameter &&
      State.Stack.back().FirstLessLess == 0)
    return true;

  if (Current.Type == TT_SelectorName &&
      State.Stack.back().ObjCSelectorNameFound &&
      State.Stack.back().BreakBeforeParameter)
    return true;
  if (Previous.ClosesTemplateDeclaration && Current.NestingLevel == 0 &&
      !Current.isTrailingComment())
    return true;

  // If the return type spans multiple lines, wrap before the function name.
  if ((Current.Type == TT_FunctionDeclarationName ||
       Current.is(tok::kw_operator)) &&
      State.Stack.back().BreakBeforeParameter)
    return true;

  if (startsSegmentOfBuilderTypeCall(Current) &&
      (State.Stack.back().CallContinuation != 0 ||
       (State.Stack.back().BreakBeforeParameter &&
        State.Stack.back().ContainsUnwrappedBuilder)))
    return true;

  // The following could be precomputed as they do not depend on the state.
  // However, as they should take effect only if the UnwrappedLine does not fit
  // into the ColumnLimit, they are checked here in the ContinuationIndenter.
  if (Style.ColumnLimit != 0 && Previous.BlockKind == BK_Block &&
      Previous.is(tok::l_brace) && !Current.isOneOf(tok::r_brace, tok::comment))
    return true;

  return false;
}

unsigned ContinuationIndenter::addTokenToState(LineState &State, bool Newline,
                                               bool DryRun,
                                               unsigned ExtraSpaces) {
  const FormatToken &Current = *State.NextToken;

  assert(!State.Stack.empty());
  if ((Current.Type == TT_ImplicitStringLiteral &&
       (Current.Previous->Tok.getIdentifierInfo() == nullptr ||
        Current.Previous->Tok.getIdentifierInfo()->getPPKeywordID() ==
            tok::pp_not_keyword))) {
    // FIXME: Is this correct?
    int WhitespaceLength = SourceMgr.getSpellingColumnNumber(
                               State.NextToken->WhitespaceRange.getEnd()) -
                           SourceMgr.getSpellingColumnNumber(
                               State.NextToken->WhitespaceRange.getBegin());
    State.Column += WhitespaceLength;
    moveStateToNextToken(State, DryRun, /*Newline=*/false);
    return 0;
  }

  unsigned Penalty = 0;
  if (Newline)
    Penalty = addTokenOnNewLine(State, DryRun);
  else
    addTokenOnCurrentLine(State, DryRun, ExtraSpaces);

  return moveStateToNextToken(State, DryRun, Newline) + Penalty;
}

void ContinuationIndenter::addTokenOnCurrentLine(LineState &State, bool DryRun,
                                                 unsigned ExtraSpaces) {
  FormatToken &Current = *State.NextToken;
  const FormatToken &Previous = *State.NextToken->Previous;
  if (Current.is(tok::equal) &&
      (State.Line->First->is(tok::kw_for) || Current.NestingLevel == 0) &&
      State.Stack.back().VariablePos == 0) {
    State.Stack.back().VariablePos = State.Column;
    // Move over * and & if they are bound to the variable name.
    const FormatToken *Tok = &Previous;
    while (Tok && State.Stack.back().VariablePos >= Tok->ColumnWidth) {
      State.Stack.back().VariablePos -= Tok->ColumnWidth;
      if (Tok->SpacesRequiredBefore != 0)
        break;
      Tok = Tok->Previous;
    }
    if (Previous.PartOfMultiVariableDeclStmt)
      State.Stack.back().LastSpace = State.Stack.back().VariablePos;
  }

  unsigned Spaces = Current.SpacesRequiredBefore + ExtraSpaces;

  if (!DryRun)
    Whitespaces.replaceWhitespace(Current, /*Newlines=*/0, /*IndentLevel=*/0,
                                  Spaces, State.Column + Spaces);

  if (Current.Type == TT_SelectorName &&
      !State.Stack.back().ObjCSelectorNameFound) {
    if (Current.LongestObjCSelectorName == 0)
      State.Stack.back().AlignColons = false;
    else if (State.Stack.back().Indent + Current.LongestObjCSelectorName >
             State.Column + Spaces + Current.ColumnWidth)
      State.Stack.back().ColonPos =
          State.Stack.back().Indent + Current.LongestObjCSelectorName;
    else
      State.Stack.back().ColonPos = State.Column + Spaces + Current.ColumnWidth;
  }

  if (Previous.opensScope() && Previous.Type != TT_ObjCMethodExpr &&
      (Current.Type != TT_LineComment || Previous.BlockKind == BK_BracedInit))
    State.Stack.back().Indent = State.Column + Spaces;
  if (State.Stack.back().AvoidBinPacking && startsNextParameter(Current, Style))
    State.Stack.back().NoLineBreak = true;
  if (startsSegmentOfBuilderTypeCall(Current))
    State.Stack.back().ContainsUnwrappedBuilder = true;

  State.Column += Spaces;
  if (Current.isNot(tok::comment) && Previous.is(tok::l_paren) &&
      Previous.Previous && Previous.Previous->isOneOf(tok::kw_if, tok::kw_for))
    // Treat the condition inside an if as if it was a second function
    // parameter, i.e. let nested calls have a continuation indent.
    State.Stack.back().LastSpace = State.Column;
  else if (!Current.isOneOf(tok::comment, tok::caret) &&
           (Previous.is(tok::comma) ||
            (Previous.is(tok::colon) && Previous.Type == TT_ObjCMethodExpr)))
    State.Stack.back().LastSpace = State.Column;
  else if ((Previous.Type == TT_BinaryOperator ||
            Previous.Type == TT_ConditionalExpr ||
            Previous.Type == TT_CtorInitializerColon) &&
           ((Previous.getPrecedence() != prec::Assignment &&
             (Previous.isNot(tok::lessless) || Previous.OperatorIndex != 0 ||
              !Previous.LastOperator)) ||
            Current.StartsBinaryExpression))
    // Always indent relative to the RHS of the expression unless this is a
    // simple assignment without binary expression on the RHS. Also indent
    // relative to unary operators and the colons of constructor initializers.
    State.Stack.back().LastSpace = State.Column;
  else if (Previous.Type == TT_InheritanceColon) {
    State.Stack.back().Indent = State.Column;
    State.Stack.back().LastSpace = State.Column;
  } else if (Previous.opensScope()) {
    // If a function has a trailing call, indent all parameters from the
    // opening parenthesis. This avoids confusing indents like:
    //   OuterFunction(InnerFunctionCall( // break
    //       ParameterToInnerFunction))   // break
    //       .SecondInnerFunctionCall();
    bool HasTrailingCall = false;
    if (Previous.MatchingParen) {
      const FormatToken *Next = Previous.MatchingParen->getNextNonComment();
      HasTrailingCall = Next && Next->isMemberAccess();
    }
    if (HasTrailingCall &&
        State.Stack[State.Stack.size() - 2].CallContinuation == 0)
      State.Stack.back().LastSpace = State.Column;
  }
}

unsigned ContinuationIndenter::addTokenOnNewLine(LineState &State,
                                                 bool DryRun) {
  FormatToken &Current = *State.NextToken;
  const FormatToken &Previous = *State.NextToken->Previous;

  // Extra penalty that needs to be added because of the way certain line
  // breaks are chosen.
  unsigned Penalty = 0;

  const FormatToken *PreviousNonComment = Current.getPreviousNonComment();
  const FormatToken *NextNonComment = Previous.getNextNonComment();
  if (!NextNonComment)
    NextNonComment = &Current;
  // The first line break on any NestingLevel causes an extra penalty in order
  // prefer similar line breaks.
  if (!State.Stack.back().ContainsLineBreak)
    Penalty += 15;
  State.Stack.back().ContainsLineBreak = true;

  Penalty += State.NextToken->SplitPenalty;

  // Breaking before the first "<<" is generally not desirable if the LHS is
  // short. Also always add the penalty if the LHS is split over mutliple lines
  // to avoid unnecessary line breaks that just work around this penalty.
  if (NextNonComment->is(tok::lessless) &&
      State.Stack.back().FirstLessLess == 0 &&
      (State.Column <= Style.ColumnLimit / 3 ||
       State.Stack.back().BreakBeforeParameter))
    Penalty += Style.PenaltyBreakFirstLessLess;

  State.Column = getNewLineColumn(State);
  if (NextNonComment->isMemberAccess()) {
    if (State.Stack.back().CallContinuation == 0)
      State.Stack.back().CallContinuation = State.Column;
  } else if (NextNonComment->Type == TT_SelectorName) {
    if (!State.Stack.back().ObjCSelectorNameFound) {
      if (NextNonComment->LongestObjCSelectorName == 0) {
        State.Stack.back().AlignColons = false;
      } else {
        State.Stack.back().ColonPos =
            State.Stack.back().Indent + NextNonComment->LongestObjCSelectorName;
      }
    } else if (State.Stack.back().AlignColons &&
               State.Stack.back().ColonPos <= NextNonComment->ColumnWidth) {
      State.Stack.back().ColonPos = State.Column + NextNonComment->ColumnWidth;
    }
  } else if (PreviousNonComment && PreviousNonComment->is(tok::colon) &&
             (PreviousNonComment->Type == TT_ObjCMethodExpr ||
              PreviousNonComment->Type == TT_DictLiteral)) {
    // FIXME: This is hacky, find a better way. The problem is that in an ObjC
    // method expression, the block should be aligned to the line starting it,
    // e.g.:
    //   [aaaaaaaaaaaaaaa aaaaaaaaa: \\ break for some reason
    //                        ^(int *i) {
    //                            // ...
    //                        }];
    // Thus, we set LastSpace of the next higher NestingLevel, to which we move
    // when we consume all of the "}"'s FakeRParens at the "{".
    if (State.Stack.size() > 1)
      State.Stack[State.Stack.size() - 2].LastSpace =
          std::max(State.Stack.back().LastSpace, State.Stack.back().Indent) +
          Style.ContinuationIndentWidth;
  }

  if ((Previous.isOneOf(tok::comma, tok::semi) &&
       !State.Stack.back().AvoidBinPacking) ||
      Previous.Type == TT_BinaryOperator)
    State.Stack.back().BreakBeforeParameter = false;
  if (Previous.Type == TT_TemplateCloser && Current.NestingLevel == 0)
    State.Stack.back().BreakBeforeParameter = false;
  if (NextNonComment->is(tok::question) ||
      (PreviousNonComment && PreviousNonComment->is(tok::question)))
    State.Stack.back().BreakBeforeParameter = true;

  if (!DryRun) {
    unsigned Newlines = std::max(
        1u, std::min(Current.NewlinesBefore, Style.MaxEmptyLinesToKeep + 1));
    Whitespaces.replaceWhitespace(Current, Newlines,
                                  State.Stack.back().IndentLevel, State.Column,
                                  State.Column, State.Line->InPPDirective);
  }

  if (!Current.isTrailingComment())
    State.Stack.back().LastSpace = State.Column;
  State.StartOfLineLevel = Current.NestingLevel;
  State.LowestLevelOnLine = Current.NestingLevel;

  // Any break on this level means that the parent level has been broken
  // and we need to avoid bin packing there.
  bool JavaScriptFormat = Style.Language == FormatStyle::LK_JavaScript &&
                          Current.is(tok::r_brace) &&
                          State.Stack.size() > 1 &&
                          State.Stack[State.Stack.size() - 2].JSFunctionInlined;
  if (!JavaScriptFormat) {
    for (unsigned i = 0, e = State.Stack.size() - 1; i != e; ++i) {
      State.Stack[i].BreakBeforeParameter = true;
    }
  }

  if (PreviousNonComment &&
      !PreviousNonComment->isOneOf(tok::comma, tok::semi) &&
      PreviousNonComment->Type != TT_TemplateCloser &&
      PreviousNonComment->Type != TT_BinaryOperator &&
      Current.Type != TT_BinaryOperator && !PreviousNonComment->opensScope())
    State.Stack.back().BreakBeforeParameter = true;

  // If we break after { or the [ of an array initializer, we should also break
  // before the corresponding } or ].
  if (PreviousNonComment &&
      (PreviousNonComment->is(tok::l_brace) ||
       PreviousNonComment->Type == TT_ArrayInitializerLSquare))
    State.Stack.back().BreakBeforeClosingBrace = true;

  if (State.Stack.back().AvoidBinPacking) {
    // If we are breaking after '(', '{', '<', this is not bin packing
    // unless AllowAllParametersOfDeclarationOnNextLine is false or this is a
    // dict/object literal.
    if (!(Previous.isOneOf(tok::l_paren, tok::l_brace) ||
          Previous.Type == TT_BinaryOperator) ||
        (!Style.AllowAllParametersOfDeclarationOnNextLine &&
         State.Line->MustBeDeclaration) ||
        Previous.Type == TT_DictLiteral)
      State.Stack.back().BreakBeforeParameter = true;
  }

  return Penalty;
}

unsigned ContinuationIndenter::getNewLineColumn(const LineState &State) {
  if (!State.NextToken || !State.NextToken->Previous)
    return 0;
  FormatToken &Current = *State.NextToken;
  const FormatToken &Previous = *State.NextToken->Previous;
  // If we are continuing an expression, we want to use the continuation indent.
  unsigned ContinuationIndent =
      std::max(State.Stack.back().LastSpace, State.Stack.back().Indent) +
      Style.ContinuationIndentWidth;
  const FormatToken *PreviousNonComment = Current.getPreviousNonComment();
  const FormatToken *NextNonComment = Previous.getNextNonComment();
  if (!NextNonComment)
    NextNonComment = &Current;
  if (NextNonComment->is(tok::l_brace) && NextNonComment->BlockKind == BK_Block)
    return Current.NestingLevel == 0 ? State.FirstIndent
                                     : State.Stack.back().Indent;
  if (Current.isOneOf(tok::r_brace, tok::r_square)) {
    if (State.Stack.size() > 1 &&
        State.Stack[State.Stack.size() - 2].JSFunctionInlined)
      return State.FirstIndent;
    if (Current.closesBlockTypeList(Style) ||
        (Current.MatchingParen &&
         Current.MatchingParen->BlockKind == BK_BracedInit))
      return State.Stack[State.Stack.size() - 2].LastSpace;
    else
      return State.FirstIndent;
  }
  if (Current.is(tok::identifier) && Current.Next &&
      Current.Next->Type == TT_DictLiteral)
    return State.Stack.back().Indent;
  if (NextNonComment->isStringLiteral() && State.StartOfStringLiteral != 0)
    return State.StartOfStringLiteral;
  if (NextNonComment->is(tok::lessless) &&
      State.Stack.back().FirstLessLess != 0)
    return State.Stack.back().FirstLessLess;
  if (NextNonComment->isMemberAccess()) {
    if (State.Stack.back().CallContinuation == 0) {
      return ContinuationIndent;
    } else {
      return State.Stack.back().CallContinuation;
    }
  }
  if (State.Stack.back().QuestionColumn != 0 &&
      ((NextNonComment->is(tok::colon) &&
        NextNonComment->Type == TT_ConditionalExpr) ||
       Previous.Type == TT_ConditionalExpr))
    return State.Stack.back().QuestionColumn;
  if (Previous.is(tok::comma) && State.Stack.back().VariablePos != 0)
    return State.Stack.back().VariablePos;
  if ((PreviousNonComment && (PreviousNonComment->ClosesTemplateDeclaration ||
                              PreviousNonComment->Type == TT_AttributeParen)) ||
      (!Style.IndentWrappedFunctionNames &&
       (NextNonComment->is(tok::kw_operator) ||
        NextNonComment->Type == TT_FunctionDeclarationName)))
    return std::max(State.Stack.back().LastSpace, State.Stack.back().Indent);
  if (NextNonComment->Type == TT_SelectorName) {
    if (!State.Stack.back().ObjCSelectorNameFound) {
      if (NextNonComment->LongestObjCSelectorName == 0) {
        return State.Stack.back().Indent;
      } else {
        return State.Stack.back().Indent +
               NextNonComment->LongestObjCSelectorName -
               NextNonComment->ColumnWidth;
      }
    } else if (!State.Stack.back().AlignColons) {
      return State.Stack.back().Indent;
    } else if (State.Stack.back().ColonPos > NextNonComment->ColumnWidth) {
      return State.Stack.back().ColonPos - NextNonComment->ColumnWidth;
    } else {
      return State.Stack.back().Indent;
    }
  }
  if (NextNonComment->Type == TT_ArraySubscriptLSquare) {
    if (State.Stack.back().StartOfArraySubscripts != 0)
      return State.Stack.back().StartOfArraySubscripts;
    else
      return ContinuationIndent;
  }
  if (NextNonComment->Type == TT_StartOfName ||
      Previous.isOneOf(tok::coloncolon, tok::equal)) {
    return ContinuationIndent;
  }
  if (PreviousNonComment && PreviousNonComment->is(tok::colon) &&
      (PreviousNonComment->Type == TT_ObjCMethodExpr ||
       PreviousNonComment->Type == TT_DictLiteral))
    return ContinuationIndent;
  if (NextNonComment->Type == TT_CtorInitializerColon)
    return State.FirstIndent + Style.ConstructorInitializerIndentWidth;
  if (NextNonComment->Type == TT_CtorInitializerComma)
    return State.Stack.back().Indent;
  if (State.Stack.back().Indent == State.FirstIndent && PreviousNonComment &&
      PreviousNonComment->isNot(tok::r_brace))
    // Ensure that we fall back to the continuation indent width instead of
    // just flushing continuations left.
    return State.Stack.back().Indent + Style.ContinuationIndentWidth;
  return State.Stack.back().Indent;
}

unsigned ContinuationIndenter::moveStateToNextToken(LineState &State,
                                                    bool DryRun, bool Newline) {
  assert(State.Stack.size());
  const FormatToken &Current = *State.NextToken;

  if (Current.Type == TT_InheritanceColon)
    State.Stack.back().AvoidBinPacking = true;
  if (Current.is(tok::lessless) && Current.Type != TT_OverloadedOperator) {
    if (State.Stack.back().FirstLessLess == 0)
      State.Stack.back().FirstLessLess = State.Column;
    else
      State.Stack.back().LastOperatorWrapped = Newline;
  }
  if ((Current.Type == TT_BinaryOperator && Current.isNot(tok::lessless)) ||
      Current.Type == TT_ConditionalExpr)
    State.Stack.back().LastOperatorWrapped = Newline;
  if (Current.Type == TT_ArraySubscriptLSquare &&
      State.Stack.back().StartOfArraySubscripts == 0)
    State.Stack.back().StartOfArraySubscripts = State.Column;
  if ((Current.is(tok::question) && Style.BreakBeforeTernaryOperators) ||
      (Current.getPreviousNonComment() && Current.isNot(tok::colon) &&
       Current.getPreviousNonComment()->is(tok::question) &&
       !Style.BreakBeforeTernaryOperators))
    State.Stack.back().QuestionColumn = State.Column;
  if (!Current.opensScope() && !Current.closesScope())
    State.LowestLevelOnLine =
        std::min(State.LowestLevelOnLine, Current.NestingLevel);
  if (Current.isMemberAccess())
    State.Stack.back().StartOfFunctionCall =
        Current.LastOperator ? 0 : State.Column + Current.ColumnWidth;
  if (Current.Type == TT_SelectorName)
    State.Stack.back().ObjCSelectorNameFound = true;
  if (Current.Type == TT_CtorInitializerColon) {
    // Indent 2 from the column, so:
    // SomeClass::SomeClass()
    //     : First(...), ...
    //       Next(...)
    //       ^ line up here.
    State.Stack.back().Indent =
        State.Column + (Style.BreakConstructorInitializersBeforeComma ? 0 : 2);
    if (Style.ConstructorInitializerAllOnOneLineOrOnePerLine)
      State.Stack.back().AvoidBinPacking = true;
    State.Stack.back().BreakBeforeParameter = false;
  }

  // In ObjC method declaration we align on the ":" of parameters, but we need
  // to ensure that we indent parameters on subsequent lines by at least our
  // continuation indent width.
  if (Current.Type == TT_ObjCMethodSpecifier)
    State.Stack.back().Indent += Style.ContinuationIndentWidth;

  // Insert scopes created by fake parenthesis.
  const FormatToken *Previous = Current.getPreviousNonComment();

  // Add special behavior to support a format commonly used for JavaScript
  // closures:
  //   SomeFunction(function() {
  //     foo();
  //     bar();
  //   }, a, b, c);
  if (Style.Language == FormatStyle::LK_JavaScript) {
    if (Current.isNot(tok::comment) && Previous && Previous->is(tok::l_brace) &&
        State.Stack.size() > 1) {
      if (State.Stack[State.Stack.size() - 2].JSFunctionInlined && Newline) {
        for (unsigned i = 0, e = State.Stack.size() - 1; i != e; ++i) {
          State.Stack[i].NoLineBreak = true;
        }
      }
      State.Stack[State.Stack.size() - 2].JSFunctionInlined = false;
    }
    if (Current.TokenText == "function")
      State.Stack.back().JSFunctionInlined = !Newline;
  }

  moveStatePastFakeLParens(State, Newline);
  moveStatePastScopeOpener(State, Newline);
  moveStatePastScopeCloser(State);
  moveStatePastFakeRParens(State);

  if (Current.isStringLiteral() && State.StartOfStringLiteral == 0) {
    State.StartOfStringLiteral = State.Column;
  } else if (!Current.isOneOf(tok::comment, tok::identifier, tok::hash) &&
             !Current.isStringLiteral()) {
    State.StartOfStringLiteral = 0;
  }

  State.Column += Current.ColumnWidth;
  State.NextToken = State.NextToken->Next;
  unsigned Penalty = breakProtrudingToken(Current, State, DryRun);
  if (State.Column > getColumnLimit(State)) {
    unsigned ExcessCharacters = State.Column - getColumnLimit(State);
    Penalty += Style.PenaltyExcessCharacter * ExcessCharacters;
  }

  if (Current.Role)
    Current.Role->formatFromToken(State, this, DryRun);
  // If the previous has a special role, let it consume tokens as appropriate.
  // It is necessary to start at the previous token for the only implemented
  // role (comma separated list). That way, the decision whether or not to break
  // after the "{" is already done and both options are tried and evaluated.
  // FIXME: This is ugly, find a better way.
  if (Previous && Previous->Role)
    Penalty += Previous->Role->formatAfterToken(State, this, DryRun);

  return Penalty;
}

void ContinuationIndenter::moveStatePastFakeLParens(LineState &State,
                                                    bool Newline) {
  const FormatToken &Current = *State.NextToken;
  const FormatToken *Previous = Current.getPreviousNonComment();

  // Don't add extra indentation for the first fake parenthesis after
  // 'return', assignments or opening <({[. The indentation for these cases
  // is special cased.
  bool SkipFirstExtraIndent =
      (Previous && (Previous->opensScope() || Previous->is(tok::kw_return) ||
                    Previous->getPrecedence() == prec::Assignment ||
                    Previous->Type == TT_ObjCMethodExpr));
  for (SmallVectorImpl<prec::Level>::const_reverse_iterator
           I = Current.FakeLParens.rbegin(),
           E = Current.FakeLParens.rend();
       I != E; ++I) {
    ParenState NewParenState = State.Stack.back();
    NewParenState.ContainsLineBreak = false;

    // Indent from 'LastSpace' unless this the fake parentheses encapsulating a
    // builder type call after 'return'. If such a call is line-wrapped, we
    // commonly just want to indent from the start of the line.
    if (!Previous || Previous->isNot(tok::kw_return) || *I > 0)
      NewParenState.Indent =
          std::max(std::max(State.Column, NewParenState.Indent),
                   State.Stack.back().LastSpace);

    // Don't allow the RHS of an operator to be split over multiple lines unless
    // there is a line-break right after the operator.
    // Exclude relational operators, as there, it is always more desirable to
    // have the LHS 'left' of the RHS.
    if (Previous && Previous->getPrecedence() > prec::Assignment &&
        (Previous->Type == TT_BinaryOperator ||
         Previous->Type == TT_ConditionalExpr) &&
        Previous->getPrecedence() != prec::Relational) {
      bool BreakBeforeOperator = Previous->is(tok::lessless) ||
                                 (Previous->Type == TT_BinaryOperator &&
                                  Style.BreakBeforeBinaryOperators) ||
                                 (Previous->Type == TT_ConditionalExpr &&
                                  Style.BreakBeforeTernaryOperators);
      if ((!Newline && !BreakBeforeOperator) ||
          (!State.Stack.back().LastOperatorWrapped && BreakBeforeOperator))
        NewParenState.NoLineBreak = true;
    }

    // Do not indent relative to the fake parentheses inserted for "." or "->".
    // This is a special case to make the following to statements consistent:
    //   OuterFunction(InnerFunctionCall( // break
    //       ParameterToInnerFunction));
    //   OuterFunction(SomeObject.InnerFunctionCall( // break
    //       ParameterToInnerFunction));
    if (*I > prec::Unknown)
      NewParenState.LastSpace = std::max(NewParenState.LastSpace, State.Column);
    NewParenState.StartOfFunctionCall = State.Column;

    // Always indent conditional expressions. Never indent expression where
    // the 'operator' is ',', ';' or an assignment (i.e. *I <=
    // prec::Assignment) as those have different indentation rules. Indent
    // other expression, unless the indentation needs to be skipped.
    if (*I == prec::Conditional ||
        (!SkipFirstExtraIndent && *I > prec::Assignment &&
         !Style.BreakBeforeBinaryOperators))
      NewParenState.Indent += Style.ContinuationIndentWidth;
    if ((Previous && !Previous->opensScope()) || *I > prec::Comma)
      NewParenState.BreakBeforeParameter = false;
    State.Stack.push_back(NewParenState);
    SkipFirstExtraIndent = false;
  }
}

// Remove the fake r_parens after 'Tok'.
static void consumeRParens(LineState& State, const FormatToken &Tok) {
  for (unsigned i = 0, e = Tok.FakeRParens; i != e; ++i) {
    unsigned VariablePos = State.Stack.back().VariablePos;
    assert(State.Stack.size() > 1);
    if (State.Stack.size() == 1) {
      // Do not pop the last element.
      break;
    }
    State.Stack.pop_back();
    State.Stack.back().VariablePos = VariablePos;
  }
}

// Returns whether 'Tok' opens or closes a scope requiring special handling
// of the subsequent fake r_parens.
//
// For example, if this is an l_brace starting a nested block, we pretend (wrt.
// to indentation) that we already consumed the corresponding r_brace. Thus, we
// remove all ParenStates caused by fake parentheses that end at the r_brace.
// The net effect of this is that we don't indent relative to the l_brace, if
// the nested block is the last parameter of a function. This formats:
//
//   SomeFunction(a, [] {
//     f();  // break
//   });
//
// instead of:
//   SomeFunction(a, [] {
//                     f();  // break
//                   });
static bool fakeRParenSpecialCase(const LineState &State) {
  const FormatToken &Tok = *State.NextToken;
  if (!Tok.MatchingParen)
    return false;
  const FormatToken *Left = &Tok;
  if (Tok.isOneOf(tok::r_brace, tok::r_square))
    Left = Tok.MatchingParen;
  return !State.Stack.back().HasMultipleNestedBlocks &&
         Left->isOneOf(tok::l_brace, tok::l_square) &&
         (Left->BlockKind == BK_Block ||
          Left->Type == TT_ArrayInitializerLSquare ||
          Left->Type == TT_DictLiteral);
}

void ContinuationIndenter::moveStatePastFakeRParens(LineState &State) {
  // Don't remove FakeRParens attached to r_braces that surround nested blocks
  // as they will have been removed early (see above).
  if (fakeRParenSpecialCase(State))
    return;

  consumeRParens(State, *State.NextToken);
}

void ContinuationIndenter::moveStatePastScopeOpener(LineState &State,
                                                    bool Newline) {
  const FormatToken &Current = *State.NextToken;
  if (!Current.opensScope())
    return;

  if (Current.MatchingParen && Current.BlockKind == BK_Block) {
    moveStateToNewBlock(State);
    return;
  }

  unsigned NewIndent;
  unsigned NewIndentLevel = State.Stack.back().IndentLevel;
  bool AvoidBinPacking;
  bool BreakBeforeParameter = false;
  if (Current.is(tok::l_brace) || Current.Type == TT_ArrayInitializerLSquare) {
    if (fakeRParenSpecialCase(State))
      consumeRParens(State, *Current.MatchingParen);

    NewIndent = State.Stack.back().LastSpace;
    if (Current.opensBlockTypeList(Style)) {
      NewIndent += Style.IndentWidth;
      NewIndent = std::min(State.Column + 2, NewIndent);
      ++NewIndentLevel;
    } else {
      NewIndent += Style.ContinuationIndentWidth;
      NewIndent = std::min(State.Column + 1, NewIndent);
    }
    const FormatToken *NextNoComment = Current.getNextNonComment();
    AvoidBinPacking = Current.Type == TT_ArrayInitializerLSquare ||
                      Current.Type == TT_DictLiteral ||
                      Style.Language == FormatStyle::LK_Proto ||
                      !Style.BinPackParameters ||
                      (NextNoComment &&
                       NextNoComment->Type == TT_DesignatedInitializerPeriod);
  } else {
    NewIndent = Style.ContinuationIndentWidth +
                std::max(State.Stack.back().LastSpace,
                         State.Stack.back().StartOfFunctionCall);
    AvoidBinPacking = !Style.BinPackParameters ||
                      (Style.ExperimentalAutoDetectBinPacking &&
                       (Current.PackingKind == PPK_OnePerLine ||
                        (!BinPackInconclusiveFunctions &&
                         Current.PackingKind == PPK_Inconclusive)));
    // If this '[' opens an ObjC call, determine whether all parameters fit
    // into one line and put one per line if they don't.
    if (Current.Type == TT_ObjCMethodExpr && Style.ColumnLimit != 0 &&
        getLengthToMatchingParen(Current) + State.Column >
            getColumnLimit(State))
      BreakBeforeParameter = true;
  }
  bool NoLineBreak = State.Stack.back().NoLineBreak ||
                     (Current.Type == TT_TemplateOpener &&
                      State.Stack.back().ContainsUnwrappedBuilder);
  State.Stack.push_back(ParenState(NewIndent, NewIndentLevel,
                                   State.Stack.back().LastSpace,
                                   AvoidBinPacking, NoLineBreak));
  State.Stack.back().BreakBeforeParameter = BreakBeforeParameter;
  State.Stack.back().HasMultipleNestedBlocks = Current.BlockParameterCount > 1;
}

void ContinuationIndenter::moveStatePastScopeCloser(LineState &State) {
  const FormatToken &Current = *State.NextToken;
  if (!Current.closesScope())
    return;

  // If we encounter a closing ), ], } or >, we can remove a level from our
  // stacks.
  if (State.Stack.size() > 1 &&
      (Current.isOneOf(tok::r_paren, tok::r_square) ||
       (Current.is(tok::r_brace) && State.NextToken != State.Line->First) ||
       State.NextToken->Type == TT_TemplateCloser))
    State.Stack.pop_back();

  if (Current.is(tok::r_square)) {
    // If this ends the array subscript expr, reset the corresponding value.
    const FormatToken *NextNonComment = Current.getNextNonComment();
    if (NextNonComment && NextNonComment->isNot(tok::l_square))
      State.Stack.back().StartOfArraySubscripts = 0;
  }
}

void ContinuationIndenter::moveStateToNewBlock(LineState &State) {
  // If we have already found more than one lambda introducers on this level, we
  // opt out of this because similarity between the lambdas is more important.
  if (fakeRParenSpecialCase(State))
    consumeRParens(State, *State.NextToken->MatchingParen);

  // For some reason, ObjC blocks are indented like continuations.
  unsigned NewIndent = State.Stack.back().LastSpace +
                       (State.NextToken->Type == TT_ObjCBlockLBrace
                            ? Style.ContinuationIndentWidth
                            : Style.IndentWidth);
  State.Stack.push_back(ParenState(
      NewIndent, /*NewIndentLevel=*/State.Stack.back().IndentLevel + 1,
      State.Stack.back().LastSpace, /*AvoidBinPacking=*/true,
      State.Stack.back().NoLineBreak));
  State.Stack.back().BreakBeforeParameter = true;
}

unsigned ContinuationIndenter::addMultilineToken(const FormatToken &Current,
                                                 LineState &State) {
  // Break before further function parameters on all levels.
  for (unsigned i = 0, e = State.Stack.size(); i != e; ++i)
    State.Stack[i].BreakBeforeParameter = true;

  unsigned ColumnsUsed = State.Column;
  // We can only affect layout of the first and the last line, so the penalty
  // for all other lines is constant, and we ignore it.
  State.Column = Current.LastLineColumnWidth;

  if (ColumnsUsed > getColumnLimit(State))
    return Style.PenaltyExcessCharacter * (ColumnsUsed - getColumnLimit(State));
  return 0;
}

static bool getRawStringLiteralPrefixPostfix(StringRef Text, StringRef &Prefix,
                                             StringRef &Postfix) {
  if (Text.startswith(Prefix = "R\"") || Text.startswith(Prefix = "uR\"") ||
      Text.startswith(Prefix = "UR\"") || Text.startswith(Prefix = "u8R\"") ||
      Text.startswith(Prefix = "LR\"")) {
    size_t ParenPos = Text.find('(');
    if (ParenPos != StringRef::npos) {
      StringRef Delimiter =
          Text.substr(Prefix.size(), ParenPos - Prefix.size());
      Prefix = Text.substr(0, ParenPos + 1);
      Postfix = Text.substr(Text.size() - 2 - Delimiter.size());
      return Postfix.front() == ')' && Postfix.back() == '"' &&
             Postfix.substr(1).startswith(Delimiter);
    }
  }
  return false;
}

unsigned ContinuationIndenter::breakProtrudingToken(const FormatToken &Current,
                                                    LineState &State,
                                                    bool DryRun) {
  // Don't break multi-line tokens other than block comments. Instead, just
  // update the state.
  if (Current.Type != TT_BlockComment && Current.IsMultiline)
    return addMultilineToken(Current, State);

  // Don't break implicit string literals.
  if (Current.Type == TT_ImplicitStringLiteral)
    return 0;

  if (!Current.isStringLiteral() && !Current.is(tok::comment))
    return 0;

  std::unique_ptr<BreakableToken> Token;
  unsigned StartColumn = State.Column - Current.ColumnWidth;
  unsigned ColumnLimit = getColumnLimit(State);

  if (Current.isStringLiteral()) {
    // Don't break string literals inside preprocessor directives (except for
    // #define directives, as their contents are stored in separate lines and
    // are not affected by this check).
    // This way we avoid breaking code with line directives and unknown
    // preprocessor directives that contain long string literals.
    if (State.Line->Type == LT_PreprocessorDirective)
      return 0;
    // Exempts unterminated string literals from line breaking. The user will
    // likely want to terminate the string before any line breaking is done.
    if (Current.IsUnterminatedLiteral)
      return 0;

    StringRef Text = Current.TokenText;
    StringRef Prefix;
    StringRef Postfix;
    bool IsNSStringLiteral = false;
    // FIXME: Handle whitespace between '_T', '(', '"..."', and ')'.
    // FIXME: Store Prefix and Suffix (or PrefixLength and SuffixLength to
    // reduce the overhead) for each FormatToken, which is a string, so that we
    // don't run multiple checks here on the hot path.
    if (Text.startswith("\"") && Current.Previous &&
        Current.Previous->is(tok::at)) {
      IsNSStringLiteral = true;
      Prefix = "@\"";
    }
    if ((Text.endswith(Postfix = "\"") &&
         (IsNSStringLiteral || Text.startswith(Prefix = "\"") ||
          Text.startswith(Prefix = "u\"") || Text.startswith(Prefix = "U\"") ||
          Text.startswith(Prefix = "u8\"") ||
          Text.startswith(Prefix = "L\""))) ||
        (Text.startswith(Prefix = "_T(\"") && Text.endswith(Postfix = "\")")) ||
        getRawStringLiteralPrefixPostfix(Text, Prefix, Postfix)) {
      Token.reset(new BreakableStringLiteral(
          Current, State.Line->Level, StartColumn, Prefix, Postfix,
          State.Line->InPPDirective, Encoding, Style));
    } else {
      return 0;
    }
  } else if (Current.Type == TT_BlockComment && Current.isTrailingComment()) {
    if (CommentPragmasRegex.match(Current.TokenText.substr(2)))
      return 0;
    Token.reset(new BreakableBlockComment(
        Current, State.Line->Level, StartColumn, Current.OriginalColumn,
        !Current.Previous, State.Line->InPPDirective, Encoding, Style));
  } else if (Current.Type == TT_LineComment &&
             (Current.Previous == nullptr ||
              Current.Previous->Type != TT_ImplicitStringLiteral)) {
    if (CommentPragmasRegex.match(Current.TokenText.substr(2)))
      return 0;
    Token.reset(new BreakableLineComment(Current, State.Line->Level,
                                         StartColumn, /*InPPDirective=*/false,
                                         Encoding, Style));
    // We don't insert backslashes when breaking line comments.
    ColumnLimit = Style.ColumnLimit;
  } else {
    return 0;
  }
  if (Current.UnbreakableTailLength >= ColumnLimit)
    return 0;

  unsigned RemainingSpace = ColumnLimit - Current.UnbreakableTailLength;
  bool BreakInserted = false;
  unsigned Penalty = 0;
  unsigned RemainingTokenColumns = 0;
  for (unsigned LineIndex = 0, EndIndex = Token->getLineCount();
       LineIndex != EndIndex; ++LineIndex) {
    if (!DryRun)
      Token->replaceWhitespaceBefore(LineIndex, Whitespaces);
    unsigned TailOffset = 0;
    RemainingTokenColumns =
        Token->getLineLengthAfterSplit(LineIndex, TailOffset, StringRef::npos);
    while (RemainingTokenColumns > RemainingSpace) {
      BreakableToken::Split Split =
          Token->getSplit(LineIndex, TailOffset, ColumnLimit);
      if (Split.first == StringRef::npos) {
        // The last line's penalty is handled in addNextStateToQueue().
        if (LineIndex < EndIndex - 1)
          Penalty += Style.PenaltyExcessCharacter *
                     (RemainingTokenColumns - RemainingSpace);
        break;
      }
      assert(Split.first != 0);
      unsigned NewRemainingTokenColumns = Token->getLineLengthAfterSplit(
          LineIndex, TailOffset + Split.first + Split.second, StringRef::npos);

      // We can remove extra whitespace instead of breaking the line.
      if (RemainingTokenColumns + 1 - Split.second <= RemainingSpace) {
        RemainingTokenColumns = 0;
        if (!DryRun)
          Token->replaceWhitespace(LineIndex, TailOffset, Split, Whitespaces);
        break;
      }

      // When breaking before a tab character, it may be moved by a few columns,
      // but will still be expanded to the next tab stop, so we don't save any
      // columns.
      if (NewRemainingTokenColumns == RemainingTokenColumns)
        break;

      assert(NewRemainingTokenColumns < RemainingTokenColumns);
      if (!DryRun)
        Token->insertBreak(LineIndex, TailOffset, Split, Whitespaces);
      Penalty += Current.SplitPenalty;
      unsigned ColumnsUsed =
          Token->getLineLengthAfterSplit(LineIndex, TailOffset, Split.first);
      if (ColumnsUsed > ColumnLimit) {
        Penalty += Style.PenaltyExcessCharacter * (ColumnsUsed - ColumnLimit);
      }
      TailOffset += Split.first + Split.second;
      RemainingTokenColumns = NewRemainingTokenColumns;
      BreakInserted = true;
    }
  }

  State.Column = RemainingTokenColumns;

  if (BreakInserted) {
    // If we break the token inside a parameter list, we need to break before
    // the next parameter on all levels, so that the next parameter is clearly
    // visible. Line comments already introduce a break.
    if (Current.Type != TT_LineComment) {
      for (unsigned i = 0, e = State.Stack.size(); i != e; ++i)
        State.Stack[i].BreakBeforeParameter = true;
    }

    Penalty += Current.isStringLiteral() ? Style.PenaltyBreakString
                                         : Style.PenaltyBreakComment;

    State.Stack.back().LastSpace = StartColumn;
  }
  return Penalty;
}

unsigned ContinuationIndenter::getColumnLimit(const LineState &State) const {
  // In preprocessor directives reserve two chars for trailing " \"
  return Style.ColumnLimit - (State.Line->InPPDirective ? 2 : 0);
}

bool ContinuationIndenter::nextIsMultilineString(const LineState &State) {
  const FormatToken &Current = *State.NextToken;
  if (!Current.isStringLiteral() || Current.Type == TT_ImplicitStringLiteral)
    return false;
  // We never consider raw string literals "multiline" for the purpose of
  // AlwaysBreakBeforeMultilineStrings implementation as they are special-cased
  // (see TokenAnnotator::mustBreakBefore().
  if (Current.TokenText.startswith("R\""))
    return false;
  if (Current.IsMultiline)
    return true;
  if (Current.getNextNonComment() &&
      Current.getNextNonComment()->isStringLiteral())
    return true; // Implicit concatenation.
  if (State.Column + Current.ColumnWidth + Current.UnbreakableTailLength >
      Style.ColumnLimit)
    return true; // String will be split.
  return false;
}

} // namespace format
} // namespace clang
