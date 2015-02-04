//===--- UnwrappedLineFormatter.cpp - Format C++ code ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UnwrappedLineFormatter.h"
#include "WhitespaceManager.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "format-formatter"

namespace clang {
namespace format {

namespace {

bool startsExternCBlock(const AnnotatedLine &Line) {
  const FormatToken *Next = Line.First->getNextNonComment();
  const FormatToken *NextNext = Next ? Next->getNextNonComment() : nullptr;
  return Line.First->is(tok::kw_extern) && Next && Next->isStringLiteral() &&
         NextNext && NextNext->is(tok::l_brace);
}

class LineJoiner {
public:
  LineJoiner(const FormatStyle &Style, const AdditionalKeywords &Keywords)
      : Style(Style), Keywords(Keywords) {}

  /// \brief Calculates how many lines can be merged into 1 starting at \p I.
  unsigned
  tryFitMultipleLinesInOne(unsigned Indent,
                           SmallVectorImpl<AnnotatedLine *>::const_iterator I,
                           SmallVectorImpl<AnnotatedLine *>::const_iterator E) {
    // We can never merge stuff if there are trailing line comments.
    const AnnotatedLine *TheLine = *I;
    if (TheLine->Last->is(TT_LineComment))
      return 0;

    if (Style.ColumnLimit > 0 && Indent > Style.ColumnLimit)
      return 0;

    unsigned Limit =
        Style.ColumnLimit == 0 ? UINT_MAX : Style.ColumnLimit - Indent;
    // If we already exceed the column limit, we set 'Limit' to 0. The different
    // tryMerge..() functions can then decide whether to still do merging.
    Limit = TheLine->Last->TotalLength > Limit
                ? 0
                : Limit - TheLine->Last->TotalLength;

    if (I + 1 == E || I[1]->Type == LT_Invalid || I[1]->First->MustBreakBefore)
      return 0;

    // FIXME: TheLine->Level != 0 might or might not be the right check to do.
    // If necessary, change to something smarter.
    bool MergeShortFunctions =
        Style.AllowShortFunctionsOnASingleLine == FormatStyle::SFS_All ||
        (Style.AllowShortFunctionsOnASingleLine == FormatStyle::SFS_Empty &&
         I[1]->First->is(tok::r_brace)) ||
        (Style.AllowShortFunctionsOnASingleLine == FormatStyle::SFS_Inline &&
         TheLine->Level != 0);

    if (TheLine->Last->is(TT_FunctionLBrace) &&
        TheLine->First != TheLine->Last) {
      return MergeShortFunctions ? tryMergeSimpleBlock(I, E, Limit) : 0;
    }
    if (TheLine->Last->is(tok::l_brace)) {
      return Style.BreakBeforeBraces == FormatStyle::BS_Attach
                 ? tryMergeSimpleBlock(I, E, Limit)
                 : 0;
    }
    if (I[1]->First->is(TT_FunctionLBrace) &&
        Style.BreakBeforeBraces != FormatStyle::BS_Attach) {
      if (I[1]->Last->is(TT_LineComment))
        return 0;

      // Check for Limit <= 2 to account for the " {".
      if (Limit <= 2 || (Style.ColumnLimit == 0 && containsMustBreak(TheLine)))
        return 0;
      Limit -= 2;

      unsigned MergedLines = 0;
      if (MergeShortFunctions) {
        MergedLines = tryMergeSimpleBlock(I + 1, E, Limit);
        // If we managed to merge the block, count the function header, which is
        // on a separate line.
        if (MergedLines > 0)
          ++MergedLines;
      }
      return MergedLines;
    }
    if (TheLine->First->is(tok::kw_if)) {
      return Style.AllowShortIfStatementsOnASingleLine
                 ? tryMergeSimpleControlStatement(I, E, Limit)
                 : 0;
    }
    if (TheLine->First->isOneOf(tok::kw_for, tok::kw_while)) {
      return Style.AllowShortLoopsOnASingleLine
                 ? tryMergeSimpleControlStatement(I, E, Limit)
                 : 0;
    }
    if (TheLine->First->isOneOf(tok::kw_case, tok::kw_default)) {
      return Style.AllowShortCaseLabelsOnASingleLine
                 ? tryMergeShortCaseLabels(I, E, Limit)
                 : 0;
    }
    if (TheLine->InPPDirective &&
        (TheLine->First->HasUnescapedNewline || TheLine->First->IsFirst)) {
      return tryMergeSimplePPDirective(I, E, Limit);
    }
    return 0;
  }

private:
  unsigned
  tryMergeSimplePPDirective(SmallVectorImpl<AnnotatedLine *>::const_iterator I,
                            SmallVectorImpl<AnnotatedLine *>::const_iterator E,
                            unsigned Limit) {
    if (Limit == 0)
      return 0;
    if (!I[1]->InPPDirective || I[1]->First->HasUnescapedNewline)
      return 0;
    if (I + 2 != E && I[2]->InPPDirective && !I[2]->First->HasUnescapedNewline)
      return 0;
    if (1 + I[1]->Last->TotalLength > Limit)
      return 0;
    return 1;
  }

  unsigned tryMergeSimpleControlStatement(
      SmallVectorImpl<AnnotatedLine *>::const_iterator I,
      SmallVectorImpl<AnnotatedLine *>::const_iterator E, unsigned Limit) {
    if (Limit == 0)
      return 0;
    if ((Style.BreakBeforeBraces == FormatStyle::BS_Allman ||
         Style.BreakBeforeBraces == FormatStyle::BS_GNU) &&
        (I[1]->First->is(tok::l_brace) && !Style.AllowShortBlocksOnASingleLine))
      return 0;
    if (I[1]->InPPDirective != (*I)->InPPDirective ||
        (I[1]->InPPDirective && I[1]->First->HasUnescapedNewline))
      return 0;
    Limit = limitConsideringMacros(I + 1, E, Limit);
    AnnotatedLine &Line = **I;
    if (Line.Last->isNot(tok::r_paren))
      return 0;
    if (1 + I[1]->Last->TotalLength > Limit)
      return 0;
    if (I[1]->First->isOneOf(tok::semi, tok::kw_if, tok::kw_for,
                             tok::kw_while, TT_LineComment))
      return 0;
    // Only inline simple if's (no nested if or else).
    if (I + 2 != E && Line.First->is(tok::kw_if) &&
        I[2]->First->is(tok::kw_else))
      return 0;
    return 1;
  }

  unsigned tryMergeShortCaseLabels(
      SmallVectorImpl<AnnotatedLine *>::const_iterator I,
      SmallVectorImpl<AnnotatedLine *>::const_iterator E, unsigned Limit) {
    if (Limit == 0 || I + 1 == E ||
        I[1]->First->isOneOf(tok::kw_case, tok::kw_default))
      return 0;
    unsigned NumStmts = 0;
    unsigned Length = 0;
    bool InPPDirective = I[0]->InPPDirective;
    for (; NumStmts < 3; ++NumStmts) {
      if (I + 1 + NumStmts == E)
        break;
      const AnnotatedLine *Line = I[1 + NumStmts];
      if (Line->InPPDirective != InPPDirective)
        break;
      if (Line->First->isOneOf(tok::kw_case, tok::kw_default, tok::r_brace))
        break;
      if (Line->First->isOneOf(tok::kw_if, tok::kw_for, tok::kw_switch,
                               tok::kw_while, tok::comment))
        return 0;
      Length += I[1 + NumStmts]->Last->TotalLength + 1; // 1 for the space.
    }
    if (NumStmts == 0 || NumStmts == 3 || Length > Limit)
      return 0;
    return NumStmts;
  }

  unsigned
  tryMergeSimpleBlock(SmallVectorImpl<AnnotatedLine *>::const_iterator I,
                      SmallVectorImpl<AnnotatedLine *>::const_iterator E,
                      unsigned Limit) {
    AnnotatedLine &Line = **I;

    // Don't merge ObjC @ keywords and methods.
    if (Style.Language != FormatStyle::LK_Java &&
        Line.First->isOneOf(tok::at, tok::minus, tok::plus))
      return 0;

    // Check that the current line allows merging. This depends on whether we
    // are in a control flow statements as well as several style flags.
    if (Line.First->isOneOf(tok::kw_else, tok::kw_case))
      return 0;
    if (Line.First->isOneOf(tok::kw_if, tok::kw_while, tok::kw_do, tok::kw_try,
                            tok::kw___try, tok::kw_catch, tok::kw___finally,
                            tok::kw_for, tok::r_brace) ||
        Line.First->is(Keywords.kw___except)) {
      if (!Style.AllowShortBlocksOnASingleLine)
        return 0;
      if (!Style.AllowShortIfStatementsOnASingleLine &&
          Line.First->is(tok::kw_if))
        return 0;
      if (!Style.AllowShortLoopsOnASingleLine &&
          Line.First->isOneOf(tok::kw_while, tok::kw_do, tok::kw_for))
        return 0;
      // FIXME: Consider an option to allow short exception handling clauses on
      // a single line.
      // FIXME: This isn't covered by tests.
      // FIXME: For catch, __except, __finally the first token on the line
      // is '}', so this isn't correct here.
      if (Line.First->isOneOf(tok::kw_try, tok::kw___try, tok::kw_catch,
                              Keywords.kw___except, tok::kw___finally))
        return 0;
    }

    FormatToken *Tok = I[1]->First;
    if (Tok->is(tok::r_brace) && !Tok->MustBreakBefore &&
        (Tok->getNextNonComment() == nullptr ||
         Tok->getNextNonComment()->is(tok::semi))) {
      // We merge empty blocks even if the line exceeds the column limit.
      Tok->SpacesRequiredBefore = 0;
      Tok->CanBreakBefore = true;
      return 1;
    } else if (Limit != 0 && Line.First->isNot(tok::kw_namespace) &&
               !startsExternCBlock(Line)) {
      // We don't merge short records.
      if (Line.First->isOneOf(tok::kw_class, tok::kw_union, tok::kw_struct))
        return 0;

      // Check that we still have three lines and they fit into the limit.
      if (I + 2 == E || I[2]->Type == LT_Invalid)
        return 0;
      Limit = limitConsideringMacros(I + 2, E, Limit);

      if (!nextTwoLinesFitInto(I, Limit))
        return 0;

      // Second, check that the next line does not contain any braces - if it
      // does, readability declines when putting it into a single line.
      if (I[1]->Last->is(TT_LineComment))
        return 0;
      do {
        if (Tok->is(tok::l_brace) && Tok->BlockKind != BK_BracedInit)
          return 0;
        Tok = Tok->Next;
      } while (Tok);

      // Last, check that the third line starts with a closing brace.
      Tok = I[2]->First;
      if (Tok->isNot(tok::r_brace))
        return 0;

      return 2;
    }
    return 0;
  }

  /// Returns the modified column limit for \p I if it is inside a macro and
  /// needs a trailing '\'.
  unsigned
  limitConsideringMacros(SmallVectorImpl<AnnotatedLine *>::const_iterator I,
                         SmallVectorImpl<AnnotatedLine *>::const_iterator E,
                         unsigned Limit) {
    if (I[0]->InPPDirective && I + 1 != E &&
        !I[1]->First->HasUnescapedNewline && !I[1]->First->is(tok::eof)) {
      return Limit < 2 ? 0 : Limit - 2;
    }
    return Limit;
  }

  bool nextTwoLinesFitInto(SmallVectorImpl<AnnotatedLine *>::const_iterator I,
                           unsigned Limit) {
    if (I[1]->First->MustBreakBefore || I[2]->First->MustBreakBefore)
      return false;
    return 1 + I[1]->Last->TotalLength + 1 + I[2]->Last->TotalLength <= Limit;
  }

  bool containsMustBreak(const AnnotatedLine *Line) {
    for (const FormatToken *Tok = Line->First; Tok; Tok = Tok->Next) {
      if (Tok->MustBreakBefore)
        return true;
    }
    return false;
  }

  const FormatStyle &Style;
  const AdditionalKeywords &Keywords;
};

class NoColumnLimitFormatter {
public:
  NoColumnLimitFormatter(ContinuationIndenter *Indenter) : Indenter(Indenter) {}

  /// \brief Formats the line starting at \p State, simply keeping all of the
  /// input's line breaking decisions.
  void format(unsigned FirstIndent, const AnnotatedLine *Line) {
    LineState State =
        Indenter->getInitialState(FirstIndent, Line, /*DryRun=*/false);
    while (State.NextToken) {
      bool Newline =
          Indenter->mustBreak(State) ||
          (Indenter->canBreak(State) && State.NextToken->NewlinesBefore > 0);
      Indenter->addTokenToState(State, Newline, /*DryRun=*/false);
    }
  }

private:
  ContinuationIndenter *Indenter;
};


static void markFinalized(FormatToken *Tok) {
  for (; Tok; Tok = Tok->Next) {
    Tok->Finalized = true;
    for (AnnotatedLine *Child : Tok->Children)
      markFinalized(Child->First);
  }
}

} // namespace

unsigned
UnwrappedLineFormatter::format(const SmallVectorImpl<AnnotatedLine *> &Lines,
                               bool DryRun, int AdditionalIndent,
                               bool FixBadIndentation) {
  LineJoiner Joiner(Style, Keywords);

  // Try to look up already computed penalty in DryRun-mode.
  std::pair<const SmallVectorImpl<AnnotatedLine *> *, unsigned> CacheKey(
      &Lines, AdditionalIndent);
  auto CacheIt = PenaltyCache.find(CacheKey);
  if (DryRun && CacheIt != PenaltyCache.end())
    return CacheIt->second;

  assert(!Lines.empty());
  unsigned Penalty = 0;
  std::vector<int> IndentForLevel;
  for (unsigned i = 0, e = Lines[0]->Level; i != e; ++i)
    IndentForLevel.push_back(Style.IndentWidth * i + AdditionalIndent);
  const AnnotatedLine *PreviousLine = nullptr;
  for (SmallVectorImpl<AnnotatedLine *>::const_iterator I = Lines.begin(),
                                                        E = Lines.end();
       I != E; ++I) {
    const AnnotatedLine &TheLine = **I;
    const FormatToken *FirstTok = TheLine.First;
    int Offset = getIndentOffset(*FirstTok);

    // Determine indent and try to merge multiple unwrapped lines.
    unsigned Indent;
    if (TheLine.InPPDirective) {
      Indent = TheLine.Level * Style.IndentWidth;
    } else {
      while (IndentForLevel.size() <= TheLine.Level)
        IndentForLevel.push_back(-1);
      IndentForLevel.resize(TheLine.Level + 1);
      Indent = getIndent(IndentForLevel, TheLine.Level);
    }
    unsigned LevelIndent = Indent;
    if (static_cast<int>(Indent) + Offset >= 0)
      Indent += Offset;

    // Merge multiple lines if possible.
    unsigned MergedLines = Joiner.tryFitMultipleLinesInOne(Indent, I, E);
    if (MergedLines > 0 && Style.ColumnLimit == 0) {
      // Disallow line merging if there is a break at the start of one of the
      // input lines.
      for (unsigned i = 0; i < MergedLines; ++i) {
        if (I[i + 1]->First->NewlinesBefore > 0)
          MergedLines = 0;
      }
    }
    if (!DryRun) {
      for (unsigned i = 0; i < MergedLines; ++i) {
        join(*I[i], *I[i + 1]);
      }
    }
    I += MergedLines;

    bool FixIndentation =
        FixBadIndentation && (LevelIndent != FirstTok->OriginalColumn);
    if (TheLine.First->is(tok::eof)) {
      if (PreviousLine && PreviousLine->Affected && !DryRun) {
        // Remove the file's trailing whitespace.
        unsigned Newlines = std::min(FirstTok->NewlinesBefore, 1u);
        Whitespaces->replaceWhitespace(*TheLine.First, Newlines,
                                       /*IndentLevel=*/0, /*Spaces=*/0,
                                       /*TargetColumn=*/0);
      }
    } else if (TheLine.Type != LT_Invalid &&
               (TheLine.Affected || FixIndentation)) {
      if (FirstTok->WhitespaceRange.isValid()) {
        if (!DryRun)
          formatFirstToken(*TheLine.First, PreviousLine, TheLine.Level, Indent,
                           TheLine.InPPDirective);
      } else {
        Indent = LevelIndent = FirstTok->OriginalColumn;
      }

      // If everything fits on a single line, just put it there.
      unsigned ColumnLimit = Style.ColumnLimit;
      if (I + 1 != E) {
        AnnotatedLine *NextLine = I[1];
        if (NextLine->InPPDirective && !NextLine->First->HasUnescapedNewline)
          ColumnLimit = getColumnLimit(TheLine.InPPDirective);
      }

      if (TheLine.Last->TotalLength + Indent <= ColumnLimit ||
          TheLine.Type == LT_ImportStatement) {
        LineState State = Indenter->getInitialState(Indent, &TheLine, DryRun);
        while (State.NextToken) {
          formatChildren(State, /*Newline=*/false, DryRun, Penalty);
          Indenter->addTokenToState(State, /*Newline=*/false, DryRun);
        }
      } else if (Style.ColumnLimit == 0) {
        // FIXME: Implement nested blocks for ColumnLimit = 0.
        NoColumnLimitFormatter Formatter(Indenter);
        if (!DryRun)
          Formatter.format(Indent, &TheLine);
      } else {
        Penalty += format(TheLine, Indent, DryRun);
      }

      if (!TheLine.InPPDirective)
        IndentForLevel[TheLine.Level] = LevelIndent;
    } else if (TheLine.ChildrenAffected) {
      format(TheLine.Children, DryRun);
    } else {
      // Format the first token if necessary, and notify the WhitespaceManager
      // about the unchanged whitespace.
      for (FormatToken *Tok = TheLine.First; Tok; Tok = Tok->Next) {
        if (Tok == TheLine.First && (Tok->NewlinesBefore > 0 || Tok->IsFirst)) {
          unsigned LevelIndent = Tok->OriginalColumn;
          if (!DryRun) {
            // Remove trailing whitespace of the previous line.
            if ((PreviousLine && PreviousLine->Affected) ||
                TheLine.LeadingEmptyLinesAffected) {
              formatFirstToken(*Tok, PreviousLine, TheLine.Level, LevelIndent,
                               TheLine.InPPDirective);
            } else {
              Whitespaces->addUntouchableToken(*Tok, TheLine.InPPDirective);
            }
          }

          if (static_cast<int>(LevelIndent) - Offset >= 0)
            LevelIndent -= Offset;
          if (Tok->isNot(tok::comment) && !TheLine.InPPDirective)
            IndentForLevel[TheLine.Level] = LevelIndent;
        } else if (!DryRun) {
          Whitespaces->addUntouchableToken(*Tok, TheLine.InPPDirective);
        }
      }
    }
    if (!DryRun)
      markFinalized(TheLine.First);
    PreviousLine = *I;
  }
  PenaltyCache[CacheKey] = Penalty;
  return Penalty;
}

unsigned UnwrappedLineFormatter::format(const AnnotatedLine &Line,
                                        unsigned FirstIndent, bool DryRun) {
  LineState State = Indenter->getInitialState(FirstIndent, &Line, DryRun);

  // If the ObjC method declaration does not fit on a line, we should format
  // it with one arg per line.
  if (State.Line->Type == LT_ObjCMethodDecl)
    State.Stack.back().BreakBeforeParameter = true;

  // Find best solution in solution space.
  return analyzeSolutionSpace(State, DryRun);
}

void UnwrappedLineFormatter::formatFirstToken(FormatToken &RootToken,
                                              const AnnotatedLine *PreviousLine,
                                              unsigned IndentLevel,
                                              unsigned Indent,
                                              bool InPPDirective) {
  unsigned Newlines =
      std::min(RootToken.NewlinesBefore, Style.MaxEmptyLinesToKeep + 1);
  // Remove empty lines before "}" where applicable.
  if (RootToken.is(tok::r_brace) &&
      (!RootToken.Next ||
       (RootToken.Next->is(tok::semi) && !RootToken.Next->Next)))
    Newlines = std::min(Newlines, 1u);
  if (Newlines == 0 && !RootToken.IsFirst)
    Newlines = 1;
  if (RootToken.IsFirst && !RootToken.HasUnescapedNewline)
    Newlines = 0;

  // Remove empty lines after "{".
  if (!Style.KeepEmptyLinesAtTheStartOfBlocks && PreviousLine &&
      PreviousLine->Last->is(tok::l_brace) &&
      PreviousLine->First->isNot(tok::kw_namespace) &&
      !startsExternCBlock(*PreviousLine))
    Newlines = 1;

  // Insert extra new line before access specifiers.
  if (PreviousLine && PreviousLine->Last->isOneOf(tok::semi, tok::r_brace) &&
      RootToken.isAccessSpecifier() && RootToken.NewlinesBefore == 1)
    ++Newlines;

  // Remove empty lines after access specifiers.
  if (PreviousLine && PreviousLine->First->isAccessSpecifier())
    Newlines = std::min(1u, Newlines);

  Whitespaces->replaceWhitespace(RootToken, Newlines, IndentLevel, Indent,
                                 Indent, InPPDirective &&
                                             !RootToken.HasUnescapedNewline);
}

/// \brief Get the indent of \p Level from \p IndentForLevel.
///
/// \p IndentForLevel must contain the indent for the level \c l
/// at \p IndentForLevel[l], or a value < 0 if the indent for
/// that level is unknown.
unsigned UnwrappedLineFormatter::getIndent(ArrayRef<int> IndentForLevel,
                                           unsigned Level) {
  if (IndentForLevel[Level] != -1)
    return IndentForLevel[Level];
  if (Level == 0)
    return 0;
  return getIndent(IndentForLevel, Level - 1) + Style.IndentWidth;
}

void UnwrappedLineFormatter::join(AnnotatedLine &A, const AnnotatedLine &B) {
  assert(!A.Last->Next);
  assert(!B.First->Previous);
  if (B.Affected)
    A.Affected = true;
  A.Last->Next = B.First;
  B.First->Previous = A.Last;
  B.First->CanBreakBefore = true;
  unsigned LengthA = A.Last->TotalLength + B.First->SpacesRequiredBefore;
  for (FormatToken *Tok = B.First; Tok; Tok = Tok->Next) {
    Tok->TotalLength += LengthA;
    A.Last = Tok;
  }
}

unsigned UnwrappedLineFormatter::analyzeSolutionSpace(LineState &InitialState,
                                                      bool DryRun) {
  std::set<LineState *, CompareLineStatePointers> Seen;

  // Increasing count of \c StateNode items we have created. This is used to
  // create a deterministic order independent of the container.
  unsigned Count = 0;
  QueueType Queue;

  // Insert start element into queue.
  StateNode *Node =
      new (Allocator.Allocate()) StateNode(InitialState, false, nullptr);
  Queue.push(QueueItem(OrderedPenalty(0, Count), Node));
  ++Count;

  unsigned Penalty = 0;

  // While not empty, take first element and follow edges.
  while (!Queue.empty()) {
    Penalty = Queue.top().first.first;
    StateNode *Node = Queue.top().second;
    if (!Node->State.NextToken) {
      DEBUG(llvm::dbgs() << "\n---\nPenalty for line: " << Penalty << "\n");
      break;
    }
    Queue.pop();

    // Cut off the analysis of certain solutions if the analysis gets too
    // complex. See description of IgnoreStackForComparison.
    if (Count > 10000)
      Node->State.IgnoreStackForComparison = true;

    if (!Seen.insert(&Node->State).second)
      // State already examined with lower penalty.
      continue;

    FormatDecision LastFormat = Node->State.NextToken->Decision;
    if (LastFormat == FD_Unformatted || LastFormat == FD_Continue)
      addNextStateToQueue(Penalty, Node, /*NewLine=*/false, &Count, &Queue);
    if (LastFormat == FD_Unformatted || LastFormat == FD_Break)
      addNextStateToQueue(Penalty, Node, /*NewLine=*/true, &Count, &Queue);
  }

  if (Queue.empty()) {
    // We were unable to find a solution, do nothing.
    // FIXME: Add diagnostic?
    DEBUG(llvm::dbgs() << "Could not find a solution.\n");
    return 0;
  }

  // Reconstruct the solution.
  if (!DryRun)
    reconstructPath(InitialState, Queue.top().second);

  DEBUG(llvm::dbgs() << "Total number of analyzed states: " << Count << "\n");
  DEBUG(llvm::dbgs() << "---\n");

  return Penalty;
}

#ifndef NDEBUG
static void printLineState(const LineState &State) {
  llvm::dbgs() << "State: ";
  for (const ParenState &P : State.Stack) {
    llvm::dbgs() << P.Indent << "|" << P.LastSpace << "|" << P.NestedBlockIndent
                 << " ";
  }
  llvm::dbgs() << State.NextToken->TokenText << "\n";
}
#endif

void UnwrappedLineFormatter::reconstructPath(LineState &State,
                                             StateNode *Current) {
  std::deque<StateNode *> Path;
  // We do not need a break before the initial token.
  while (Current->Previous) {
    Path.push_front(Current);
    Current = Current->Previous;
  }
  for (std::deque<StateNode *>::iterator I = Path.begin(), E = Path.end();
       I != E; ++I) {
    unsigned Penalty = 0;
    formatChildren(State, (*I)->NewLine, /*DryRun=*/false, Penalty);
    Penalty += Indenter->addTokenToState(State, (*I)->NewLine, false);

    DEBUG({
      printLineState((*I)->Previous->State);
      if ((*I)->NewLine) {
        llvm::dbgs() << "Penalty for placing "
                     << (*I)->Previous->State.NextToken->Tok.getName() << ": "
                     << Penalty << "\n";
      }
    });
  }
}

void UnwrappedLineFormatter::addNextStateToQueue(unsigned Penalty,
                                                 StateNode *PreviousNode,
                                                 bool NewLine, unsigned *Count,
                                                 QueueType *Queue) {
  if (NewLine && !Indenter->canBreak(PreviousNode->State))
    return;
  if (!NewLine && Indenter->mustBreak(PreviousNode->State))
    return;

  StateNode *Node = new (Allocator.Allocate())
      StateNode(PreviousNode->State, NewLine, PreviousNode);
  if (!formatChildren(Node->State, NewLine, /*DryRun=*/true, Penalty))
    return;

  Penalty += Indenter->addTokenToState(Node->State, NewLine, true);

  Queue->push(QueueItem(OrderedPenalty(Penalty, *Count), Node));
  ++(*Count);
}

bool UnwrappedLineFormatter::formatChildren(LineState &State, bool NewLine,
                                            bool DryRun, unsigned &Penalty) {
  const FormatToken *LBrace = State.NextToken->getPreviousNonComment();
  FormatToken &Previous = *State.NextToken->Previous;
  if (!LBrace || LBrace->isNot(tok::l_brace) || LBrace->BlockKind != BK_Block ||
      Previous.Children.size() == 0)
    // The previous token does not open a block. Nothing to do. We don't
    // assert so that we can simply call this function for all tokens.
    return true;

  if (NewLine) {
    int AdditionalIndent = State.Stack.back().Indent -
                           Previous.Children[0]->Level * Style.IndentWidth;

    Penalty += format(Previous.Children, DryRun, AdditionalIndent,
                      /*FixBadIndentation=*/true);
    return true;
  }

  if (Previous.Children[0]->First->MustBreakBefore)
    return false;

  // Cannot merge multiple statements into a single line.
  if (Previous.Children.size() > 1)
    return false;

  // Cannot merge into one line if this line ends on a comment.
  if (Previous.is(tok::comment))
    return false;

  // We can't put the closing "}" on a line with a trailing comment.
  if (Previous.Children[0]->Last->isTrailingComment())
    return false;

  // If the child line exceeds the column limit, we wouldn't want to merge it.
  // We add +2 for the trailing " }".
  if (Style.ColumnLimit > 0 &&
      Previous.Children[0]->Last->TotalLength + State.Column + 2 >
          Style.ColumnLimit)
    return false;

  if (!DryRun) {
    Whitespaces->replaceWhitespace(
        *Previous.Children[0]->First,
        /*Newlines=*/0, /*IndentLevel=*/0, /*Spaces=*/1,
        /*StartOfTokenColumn=*/State.Column, State.Line->InPPDirective);
  }
  Penalty += format(*Previous.Children[0], State.Column + 1, DryRun);

  State.Column += 1 + Previous.Children[0]->Last->TotalLength;
  return true;
}

} // namespace format
} // namespace clang
