//===--- WhitespaceManager.cpp - Format C++ code --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements WhitespaceManager class.
///
//===----------------------------------------------------------------------===//

#include "WhitespaceManager.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {
namespace format {

bool
WhitespaceManager::Change::IsBeforeInFile::operator()(const Change &C1,
                                                      const Change &C2) const {
  return SourceMgr.isBeforeInTranslationUnit(
      C1.OriginalWhitespaceRange.getBegin(),
      C2.OriginalWhitespaceRange.getBegin());
}

WhitespaceManager::Change::Change(
    bool CreateReplacement, const SourceRange &OriginalWhitespaceRange,
    unsigned IndentLevel, unsigned Spaces, unsigned StartOfTokenColumn,
    unsigned NewlinesBefore, StringRef PreviousLinePostfix,
    StringRef CurrentLinePrefix, tok::TokenKind Kind, bool ContinuesPPDirective)
    : CreateReplacement(CreateReplacement),
      OriginalWhitespaceRange(OriginalWhitespaceRange),
      StartOfTokenColumn(StartOfTokenColumn), NewlinesBefore(NewlinesBefore),
      PreviousLinePostfix(PreviousLinePostfix),
      CurrentLinePrefix(CurrentLinePrefix), Kind(Kind),
      ContinuesPPDirective(ContinuesPPDirective), IndentLevel(IndentLevel),
      Spaces(Spaces) {}

void WhitespaceManager::reset() {
  Changes.clear();
  Replaces.clear();
}

void WhitespaceManager::replaceWhitespace(FormatToken &Tok, unsigned Newlines,
                                          unsigned IndentLevel, unsigned Spaces,
                                          unsigned StartOfTokenColumn,
                                          bool InPPDirective) {
  if (Tok.Finalized)
    return;
  Tok.Decision = (Newlines > 0) ? FD_Break : FD_Continue;
  Changes.push_back(Change(true, Tok.WhitespaceRange, IndentLevel, Spaces,
                           StartOfTokenColumn, Newlines, "", "",
                           Tok.Tok.getKind(), InPPDirective && !Tok.IsFirst));
}

void WhitespaceManager::addUntouchableToken(const FormatToken &Tok,
                                            bool InPPDirective) {
  if (Tok.Finalized)
    return;
  Changes.push_back(Change(false, Tok.WhitespaceRange, /*IndentLevel=*/0,
                           /*Spaces=*/0, Tok.OriginalColumn, Tok.NewlinesBefore,
                           "", "", Tok.Tok.getKind(),
                           InPPDirective && !Tok.IsFirst));
}

void WhitespaceManager::replaceWhitespaceInToken(
    const FormatToken &Tok, unsigned Offset, unsigned ReplaceChars,
    StringRef PreviousPostfix, StringRef CurrentPrefix, bool InPPDirective,
    unsigned Newlines, unsigned IndentLevel, unsigned Spaces) {
  if (Tok.Finalized)
    return;
  Changes.push_back(Change(
      true, SourceRange(Tok.getStartOfNonWhitespace().getLocWithOffset(Offset),
                        Tok.getStartOfNonWhitespace().getLocWithOffset(
                            Offset + ReplaceChars)),
      IndentLevel, Spaces, Spaces, Newlines, PreviousPostfix, CurrentPrefix,
      // If we don't add a newline this change doesn't start a comment. Thus,
      // when we align line comments, we don't need to treat this change as one.
      // FIXME: We still need to take this change in account to properly
      // calculate the new length of the comment and to calculate the changes
      // for which to do the alignment when aligning comments.
      Tok.Type == TT_LineComment && Newlines > 0 ? tok::comment : tok::unknown,
      InPPDirective && !Tok.IsFirst));
}

const tooling::Replacements &WhitespaceManager::generateReplacements() {
  if (Changes.empty())
    return Replaces;

  std::sort(Changes.begin(), Changes.end(), Change::IsBeforeInFile(SourceMgr));
  calculateLineBreakInformation();
  alignTrailingComments();
  alignEscapedNewlines();
  generateChanges();

  return Replaces;
}

void WhitespaceManager::calculateLineBreakInformation() {
  Changes[0].PreviousEndOfTokenColumn = 0;
  for (unsigned i = 1, e = Changes.size(); i != e; ++i) {
    unsigned OriginalWhitespaceStart =
        SourceMgr.getFileOffset(Changes[i].OriginalWhitespaceRange.getBegin());
    unsigned PreviousOriginalWhitespaceEnd = SourceMgr.getFileOffset(
        Changes[i - 1].OriginalWhitespaceRange.getEnd());
    Changes[i - 1].TokenLength = OriginalWhitespaceStart -
                                 PreviousOriginalWhitespaceEnd +
                                 Changes[i].PreviousLinePostfix.size() +
                                 Changes[i - 1].CurrentLinePrefix.size();

    Changes[i].PreviousEndOfTokenColumn =
        Changes[i - 1].StartOfTokenColumn + Changes[i - 1].TokenLength;

    Changes[i - 1].IsTrailingComment =
        (Changes[i].NewlinesBefore > 0 || Changes[i].Kind == tok::eof) &&
        Changes[i - 1].Kind == tok::comment;
  }
  // FIXME: The last token is currently not always an eof token; in those
  // cases, setting TokenLength of the last token to 0 is wrong.
  Changes.back().TokenLength = 0;
  Changes.back().IsTrailingComment = Changes.back().Kind == tok::comment;
}

void WhitespaceManager::alignTrailingComments() {
  unsigned MinColumn = 0;
  unsigned MaxColumn = UINT_MAX;
  unsigned StartOfSequence = 0;
  bool BreakBeforeNext = false;
  unsigned Newlines = 0;
  for (unsigned i = 0, e = Changes.size(); i != e; ++i) {
    unsigned ChangeMinColumn = Changes[i].StartOfTokenColumn;
    // FIXME: Correctly handle ChangeMaxColumn in PP directives.
    unsigned ChangeMaxColumn = Style.ColumnLimit - Changes[i].TokenLength;
    Newlines += Changes[i].NewlinesBefore;
    if (Changes[i].IsTrailingComment) {
      // If this comment follows an } in column 0, it probably documents the
      // closing of a namespace and we don't want to align it.
      bool FollowsRBraceInColumn0 = i > 0 && Changes[i].NewlinesBefore == 0 &&
                                    Changes[i - 1].Kind == tok::r_brace &&
                                    Changes[i - 1].StartOfTokenColumn == 0;
      bool WasAlignedWithStartOfNextLine = false;
      if (Changes[i].NewlinesBefore == 1) { // A comment on its own line.
        for (unsigned j = i + 1; j != e; ++j) {
          if (Changes[j].Kind != tok::comment) { // Skip over comments.
            // The start of the next token was previously aligned with the
            // start of this comment.
            WasAlignedWithStartOfNextLine =
                (SourceMgr.getSpellingColumnNumber(
                     Changes[i].OriginalWhitespaceRange.getEnd()) ==
                 SourceMgr.getSpellingColumnNumber(
                     Changes[j].OriginalWhitespaceRange.getEnd()));
            break;
          }
        }
      }
      if (!Style.AlignTrailingComments || FollowsRBraceInColumn0) {
        alignTrailingComments(StartOfSequence, i, MinColumn);
        MinColumn = ChangeMinColumn;
        MaxColumn = ChangeMinColumn;
        StartOfSequence = i;
      } else if (BreakBeforeNext || Newlines > 1 ||
                 (ChangeMinColumn > MaxColumn || ChangeMaxColumn < MinColumn) ||
                 // Break the comment sequence if the previous line did not end
                 // in a trailing comment.
                 (Changes[i].NewlinesBefore == 1 && i > 0 &&
                  !Changes[i - 1].IsTrailingComment) ||
                 WasAlignedWithStartOfNextLine) {
        alignTrailingComments(StartOfSequence, i, MinColumn);
        MinColumn = ChangeMinColumn;
        MaxColumn = ChangeMaxColumn;
        StartOfSequence = i;
      } else {
        MinColumn = std::max(MinColumn, ChangeMinColumn);
        MaxColumn = std::min(MaxColumn, ChangeMaxColumn);
      }
      BreakBeforeNext =
          (i == 0) || (Changes[i].NewlinesBefore > 1) ||
          // Never start a sequence with a comment at the beginning of
          // the line.
          (Changes[i].NewlinesBefore == 1 && StartOfSequence == i);
      Newlines = 0;
    }
  }
  alignTrailingComments(StartOfSequence, Changes.size(), MinColumn);
}

void WhitespaceManager::alignTrailingComments(unsigned Start, unsigned End,
                                              unsigned Column) {
  for (unsigned i = Start; i != End; ++i) {
    if (Changes[i].IsTrailingComment) {
      assert(Column >= Changes[i].StartOfTokenColumn);
      Changes[i].Spaces += Column - Changes[i].StartOfTokenColumn;
      Changes[i].StartOfTokenColumn = Column;
    }
  }
}

void WhitespaceManager::alignEscapedNewlines() {
  unsigned MaxEndOfLine =
      Style.AlignEscapedNewlinesLeft ? 0 : Style.ColumnLimit;
  unsigned StartOfMacro = 0;
  for (unsigned i = 1, e = Changes.size(); i < e; ++i) {
    Change &C = Changes[i];
    if (C.NewlinesBefore > 0) {
      if (C.ContinuesPPDirective) {
        MaxEndOfLine = std::max(C.PreviousEndOfTokenColumn + 2, MaxEndOfLine);
      } else {
        alignEscapedNewlines(StartOfMacro + 1, i, MaxEndOfLine);
        MaxEndOfLine = Style.AlignEscapedNewlinesLeft ? 0 : Style.ColumnLimit;
        StartOfMacro = i;
      }
    }
  }
  alignEscapedNewlines(StartOfMacro + 1, Changes.size(), MaxEndOfLine);
}

void WhitespaceManager::alignEscapedNewlines(unsigned Start, unsigned End,
                                             unsigned Column) {
  for (unsigned i = Start; i < End; ++i) {
    Change &C = Changes[i];
    if (C.NewlinesBefore > 0) {
      assert(C.ContinuesPPDirective);
      if (C.PreviousEndOfTokenColumn + 1 > Column)
        C.EscapedNewlineColumn = 0;
      else
        C.EscapedNewlineColumn = Column;
    }
  }
}

void WhitespaceManager::generateChanges() {
  for (unsigned i = 0, e = Changes.size(); i != e; ++i) {
    const Change &C = Changes[i];
    if (C.CreateReplacement) {
      std::string ReplacementText = C.PreviousLinePostfix;
      if (C.ContinuesPPDirective)
        appendNewlineText(ReplacementText, C.NewlinesBefore,
                          C.PreviousEndOfTokenColumn, C.EscapedNewlineColumn);
      else
        appendNewlineText(ReplacementText, C.NewlinesBefore);
      appendIndentText(ReplacementText, C.IndentLevel, C.Spaces,
                       C.StartOfTokenColumn - C.Spaces);
      ReplacementText.append(C.CurrentLinePrefix);
      storeReplacement(C.OriginalWhitespaceRange, ReplacementText);
    }
  }
}

void WhitespaceManager::storeReplacement(const SourceRange &Range,
                                         StringRef Text) {
  unsigned WhitespaceLength = SourceMgr.getFileOffset(Range.getEnd()) -
                              SourceMgr.getFileOffset(Range.getBegin());
  // Don't create a replacement, if it does not change anything.
  if (StringRef(SourceMgr.getCharacterData(Range.getBegin()),
                WhitespaceLength) == Text)
    return;
  Replaces.insert(tooling::Replacement(
      SourceMgr, CharSourceRange::getCharRange(Range), Text));
}

void WhitespaceManager::appendNewlineText(std::string &Text,
                                          unsigned Newlines) {
  for (unsigned i = 0; i < Newlines; ++i)
    Text.append(UseCRLF ? "\r\n" : "\n");
}

void WhitespaceManager::appendNewlineText(std::string &Text, unsigned Newlines,
                                          unsigned PreviousEndOfTokenColumn,
                                          unsigned EscapedNewlineColumn) {
  if (Newlines > 0) {
    unsigned Offset =
        std::min<int>(EscapedNewlineColumn - 1, PreviousEndOfTokenColumn);
    for (unsigned i = 0; i < Newlines; ++i) {
      Text.append(std::string(EscapedNewlineColumn - Offset - 1, ' '));
      Text.append(UseCRLF ? "\\\r\n" : "\\\n");
      Offset = 0;
    }
  }
}

void WhitespaceManager::appendIndentText(std::string &Text,
                                         unsigned IndentLevel, unsigned Spaces,
                                         unsigned WhitespaceStartColumn) {
  switch (Style.UseTab) {
  case FormatStyle::UT_Never:
    Text.append(std::string(Spaces, ' '));
    break;
  case FormatStyle::UT_Always: {
    unsigned FirstTabWidth =
        Style.TabWidth - WhitespaceStartColumn % Style.TabWidth;
    // Indent with tabs only when there's at least one full tab.
    if (FirstTabWidth + Style.TabWidth <= Spaces) {
      Spaces -= FirstTabWidth;
      Text.append("\t");
    }
    Text.append(std::string(Spaces / Style.TabWidth, '\t'));
    Text.append(std::string(Spaces % Style.TabWidth, ' '));
    break;
  }
  case FormatStyle::UT_ForIndentation:
    if (WhitespaceStartColumn == 0) {
      unsigned Indentation = IndentLevel * Style.IndentWidth;
      // This happens, e.g. when a line in a block comment is indented less than
      // the first one.
      if (Indentation > Spaces)
        Indentation = Spaces;
      unsigned Tabs = Indentation / Style.TabWidth;
      Text.append(std::string(Tabs, '\t'));
      Spaces -= Tabs * Style.TabWidth;
    }
    Text.append(std::string(Spaces, ' '));
    break;
  }
}

} // namespace format
} // namespace clang
