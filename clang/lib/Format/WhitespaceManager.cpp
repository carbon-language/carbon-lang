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
    unsigned Spaces, unsigned StartOfTokenColumn, unsigned NewlinesBefore,
    StringRef PreviousLinePostfix, StringRef CurrentLinePrefix,
    tok::TokenKind Kind, bool ContinuesPPDirective)
    : CreateReplacement(CreateReplacement),
      OriginalWhitespaceRange(OriginalWhitespaceRange),
      StartOfTokenColumn(StartOfTokenColumn), NewlinesBefore(NewlinesBefore),
      PreviousLinePostfix(PreviousLinePostfix),
      CurrentLinePrefix(CurrentLinePrefix), Kind(Kind),
      ContinuesPPDirective(ContinuesPPDirective), Spaces(Spaces) {}

void WhitespaceManager::replaceWhitespace(const AnnotatedToken &Tok,
                                          unsigned Newlines, unsigned Spaces,
                                          unsigned StartOfTokenColumn,
                                          bool InPPDirective) {
  Changes.push_back(
      Change(true, Tok.FormatTok->WhitespaceRange, Spaces, StartOfTokenColumn,
             Newlines, "", "", Tok.FormatTok->Tok.getKind(),
             InPPDirective && !Tok.FormatTok->IsFirst));
}

void WhitespaceManager::addUntouchableToken(const FormatToken &Tok,
                                            bool InPPDirective) {
  Changes.push_back(
      Change(false, Tok.WhitespaceRange, /*Spaces=*/0,
             SourceMgr.getSpellingColumnNumber(Tok.Tok.getLocation()) - 1,
             Tok.NewlinesBefore, "", "", Tok.Tok.getKind(),
             InPPDirective && !Tok.IsFirst));
}

void WhitespaceManager::breakToken(const FormatToken &Tok, unsigned Offset,
                                   unsigned ReplaceChars,
                                   StringRef PreviousPostfix,
                                   StringRef CurrentPrefix, bool InPPDirective,
                                   unsigned Spaces) {
  Changes.push_back(Change(
      true, SourceRange(Tok.getStartOfNonWhitespace().getLocWithOffset(Offset),
                        Tok.getStartOfNonWhitespace().getLocWithOffset(
                            Offset + ReplaceChars)),
      Spaces, Spaces, 1, PreviousPostfix, CurrentPrefix,
      // FIXME: Unify token adjustment, so we don't split it between
      // BreakableToken and the WhitespaceManager. That would also allow us to
      // correctly store a tok::TokenKind instead of rolling our own enum.
      tok::unknown, InPPDirective && !Tok.IsFirst));
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
    Changes[i - 1].TokenLength =
        OriginalWhitespaceStart - PreviousOriginalWhitespaceEnd +
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
      bool WasAlignedWithStartOfNextLine =
          // A comment on its own line.
          Changes[i].NewlinesBefore == 1 &&
          // Not the last line.
          i + 1 != e &&
          // The start of the next token was previously aligned with
          // the start of this comment.
          (SourceMgr.getSpellingColumnNumber(
               Changes[i].OriginalWhitespaceRange.getEnd()) ==
           SourceMgr.getSpellingColumnNumber(
               Changes[i + 1].OriginalWhitespaceRange.getEnd())) &&
          // Which is not a comment itself.
          Changes[i + 1].Kind != tok::comment;
      if (BreakBeforeNext || Newlines > 1 ||
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
  unsigned MaxEndOfLine = 0;
  unsigned StartOfMacro = 0;
  for (unsigned i = 1, e = Changes.size(); i < e; ++i) {
    Change &C = Changes[i];
    if (C.NewlinesBefore > 0) {
      if (C.ContinuesPPDirective) {
        if (Style.AlignEscapedNewlinesLeft)
          MaxEndOfLine = std::max(C.PreviousEndOfTokenColumn + 2, MaxEndOfLine);
        else
          MaxEndOfLine = Style.ColumnLimit;
      } else {
        alignEscapedNewlines(StartOfMacro + 1, i, MaxEndOfLine);
        MaxEndOfLine = 0;
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
      std::string ReplacementText =
          C.PreviousLinePostfix +
          (C.ContinuesPPDirective
               ? getNewLineText(C.NewlinesBefore, C.Spaces,
                                C.PreviousEndOfTokenColumn,
                                C.EscapedNewlineColumn)
               : getNewLineText(C.NewlinesBefore, C.Spaces)) +
          C.CurrentLinePrefix;
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
                WhitespaceLength) ==
      Text)
    return;
  Replaces.insert(tooling::Replacement(
      SourceMgr, CharSourceRange::getCharRange(Range), Text));
}

std::string WhitespaceManager::getNewLineText(unsigned NewLines,
                                              unsigned Spaces) {
  return std::string(NewLines, '\n') + getIndentText(Spaces);
}

std::string WhitespaceManager::getNewLineText(unsigned NewLines,
                                              unsigned Spaces,
                                              unsigned PreviousEndOfTokenColumn,
                                              unsigned EscapedNewlineColumn) {
  std::string NewLineText;
  if (NewLines > 0) {
    unsigned Offset =
        std::min<int>(EscapedNewlineColumn - 1, PreviousEndOfTokenColumn);
    for (unsigned i = 0; i < NewLines; ++i) {
      NewLineText += std::string(EscapedNewlineColumn - Offset - 1, ' ');
      NewLineText += "\\\n";
      Offset = 0;
    }
  }
  return NewLineText + getIndentText(Spaces);
}

std::string WhitespaceManager::getIndentText(unsigned Spaces) {
  if (!Style.UseTab)
    return std::string(Spaces, ' ');

  return std::string(Spaces / Style.IndentWidth, '\t') +
         std::string(Spaces % Style.IndentWidth, ' ');
}

} // namespace format
} // namespace clang
