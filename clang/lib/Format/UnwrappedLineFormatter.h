//===--- UnwrappedLineFormatter.h - Format C++ code -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Implements a combinartorial exploration of all the different
/// linebreaks unwrapped lines can be formatted in.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_FORMAT_UNWRAPPEDLINEFORMATTER_H
#define LLVM_CLANG_LIB_FORMAT_UNWRAPPEDLINEFORMATTER_H

#include "ContinuationIndenter.h"
#include "clang/Format/Format.h"
#include <map>
#include <queue>
#include <string>

namespace clang {
namespace format {

class ContinuationIndenter;
class WhitespaceManager;

class UnwrappedLineFormatter {
public:
  UnwrappedLineFormatter(ContinuationIndenter *Indenter,
                         WhitespaceManager *Whitespaces,
                         const FormatStyle &Style,
                         const AdditionalKeywords &Keywords)
      : Indenter(Indenter), Whitespaces(Whitespaces), Style(Style),
        Keywords(Keywords) {}

  unsigned format(const SmallVectorImpl<AnnotatedLine *> &Lines, bool DryRun,
                  int AdditionalIndent = 0, bool FixBadIndentation = false);

private:
  /// \brief Formats an \c AnnotatedLine and returns the penalty.
  ///
  /// If \p DryRun is \c false, directly applies the changes.
  unsigned format(const AnnotatedLine &Line, unsigned FirstIndent,
                  bool DryRun);

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

  /// \brief Get the offset of the line relatively to the level.
  ///
  /// For example, 'public:' labels in classes are offset by 1 or 2
  /// characters to the left from their level.
  int getIndentOffset(const FormatToken &RootToken) {
    if (Style.Language == FormatStyle::LK_Java)
      return 0;
    if (RootToken.isAccessSpecifier(false) || RootToken.isObjCAccessSpecifier())
      return Style.AccessModifierOffset;
    return 0;
  }

  /// \brief Add a new line and the required indent before the first Token
  /// of the \c UnwrappedLine if there was no structural parsing error.
  void formatFirstToken(FormatToken &RootToken,
                        const AnnotatedLine *PreviousLine, unsigned IndentLevel,
                        unsigned Indent, bool InPPDirective);

  /// \brief Get the indent of \p Level from \p IndentForLevel.
  ///
  /// \p IndentForLevel must contain the indent for the level \c l
  /// at \p IndentForLevel[l], or a value < 0 if the indent for
  /// that level is unknown.
  unsigned getIndent(ArrayRef<int> IndentForLevel, unsigned Level);

  void join(AnnotatedLine &A, const AnnotatedLine &B);

  unsigned getColumnLimit(bool InPPDirective) const {
    // In preprocessor directives reserve two chars for trailing " \"
    return Style.ColumnLimit - (InPPDirective ? 2 : 0);
  }

  struct CompareLineStatePointers {
    bool operator()(LineState *obj1, LineState *obj2) const {
      return *obj1 < *obj2;
    }
  };

  /// \brief Analyze the entire solution space starting from \p InitialState.
  ///
  /// This implements a variant of Dijkstra's algorithm on the graph that spans
  /// the solution space (\c LineStates are the nodes). The algorithm tries to
  /// find the shortest path (the one with lowest penalty) from \p InitialState
  /// to a state where all tokens are placed. Returns the penalty.
  ///
  /// If \p DryRun is \c false, directly applies the changes.
  unsigned analyzeSolutionSpace(LineState &InitialState, bool DryRun = false);

  void reconstructPath(LineState &State, StateNode *Current);

  /// \brief Add the following state to the analysis queue \c Queue.
  ///
  /// Assume the current state is \p PreviousNode and has been reached with a
  /// penalty of \p Penalty. Insert a line break if \p NewLine is \c true.
  void addNextStateToQueue(unsigned Penalty, StateNode *PreviousNode,
                           bool NewLine, unsigned *Count, QueueType *Queue);

  /// \brief If the \p State's next token is an r_brace closing a nested block,
  /// format the nested block before it.
  ///
  /// Returns \c true if all children could be placed successfully and adapts
  /// \p Penalty as well as \p State. If \p DryRun is false, also directly
  /// creates changes using \c Whitespaces.
  ///
  /// The crucial idea here is that children always get formatted upon
  /// encountering the closing brace right after the nested block. Now, if we
  /// are currently trying to keep the "}" on the same line (i.e. \p NewLine is
  /// \c false), the entire block has to be kept on the same line (which is only
  /// possible if it fits on the line, only contains a single statement, etc.
  ///
  /// If \p NewLine is true, we format the nested block on separate lines, i.e.
  /// break after the "{", format all lines with correct indentation and the put
  /// the closing "}" on yet another new line.
  ///
  /// This enables us to keep the simple structure of the
  /// \c UnwrappedLineFormatter, where we only have two options for each token:
  /// break or don't break.
  bool formatChildren(LineState &State, bool NewLine, bool DryRun,
                      unsigned &Penalty);

  ContinuationIndenter *Indenter;
  WhitespaceManager *Whitespaces;
  FormatStyle Style;
  const AdditionalKeywords &Keywords;

  llvm::SpecificBumpPtrAllocator<StateNode> Allocator;

  // Cache to store the penalty of formatting a vector of AnnotatedLines
  // starting from a specific additional offset. Improves performance if there
  // are many nested blocks.
  std::map<std::pair<const SmallVectorImpl<AnnotatedLine *> *, unsigned>,
           unsigned> PenaltyCache;
};
} // end namespace format
} // end namespace clang

#endif // LLVM_CLANG_LIB_FORMAT_UNWRAPPEDLINEFORMATTER_H
