//===--- BreakableToken.cpp - Format C++ code -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Contains implementation of BreakableToken class and classes derived
/// from it.
///
//===----------------------------------------------------------------------===//

#include "BreakableToken.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

namespace clang {
namespace format {

BreakableToken::Split BreakableComment::getSplit(unsigned LineIndex,
                                                 unsigned TailOffset,
                                                 unsigned ColumnLimit) const {
  StringRef Text = getLine(LineIndex).substr(TailOffset);
  unsigned ContentStartColumn = getContentStartColumn(LineIndex, TailOffset);
  if (ColumnLimit <= ContentStartColumn + 1)
    return Split(StringRef::npos, 0);

  unsigned MaxSplit = ColumnLimit - ContentStartColumn + 1;
  StringRef::size_type SpaceOffset = Text.rfind(' ', MaxSplit);
  if (SpaceOffset == StringRef::npos ||
      Text.find_last_not_of(' ', SpaceOffset) == StringRef::npos) {
    SpaceOffset = Text.find(' ', MaxSplit);
  }
  if (SpaceOffset != StringRef::npos && SpaceOffset != 0) {
    StringRef BeforeCut = Text.substr(0, SpaceOffset).rtrim();
    StringRef AfterCut = Text.substr(SpaceOffset).ltrim();
    return BreakableToken::Split(BeforeCut.size(),
                                 AfterCut.begin() - BeforeCut.end());
  }
  return BreakableToken::Split(StringRef::npos, 0);
}

void BreakableComment::insertBreak(unsigned LineIndex, unsigned TailOffset,
                                   Split Split, bool InPPDirective,
                                   WhitespaceManager &Whitespaces) {
  StringRef Text = getLine(LineIndex).substr(TailOffset);
  StringRef AdditionalPrefix = Decoration;
  if (Text.size() == Split.first + Split.second) {
    // For all but the last line handle trailing space in trimLine.
    if (LineIndex < Lines.size() - 1)
      return;
    // For the last line we need to break before "*/", but not to add "* ".
    AdditionalPrefix = "";
  }

  unsigned BreakOffset = Text.data() - TokenText.data() + Split.first;
  unsigned CharsToRemove = Split.second;
  Whitespaces.breakToken(Tok, BreakOffset, CharsToRemove, "", AdditionalPrefix,
                         InPPDirective, IndentAtLineBreak);
}

BreakableBlockComment::BreakableBlockComment(const SourceManager &SourceMgr,
                                             const AnnotatedToken &Token,
                                             unsigned StartColumn)
    : BreakableComment(SourceMgr, Token.FormatTok, StartColumn + 2) {
  assert(TokenText.startswith("/*") && TokenText.endswith("*/"));

  OriginalStartColumn =
      SourceMgr.getSpellingColumnNumber(Tok.getStartOfNonWhitespace()) - 1;

  TokenText.substr(2, TokenText.size() - 4).split(Lines, "\n");

  bool NeedsStar = true;
  CommonPrefixLength = UINT_MAX;
  if (Lines.size() == 1) {
    if (Token.Parent == 0) {
      // Standalone block comments will be aligned and prefixed with *s.
      CommonPrefixLength = OriginalStartColumn + 1;
    } else {
      // Trailing comments can start on arbitrary column, and available
      // horizontal space can be too small to align consecutive lines with
      // the first one. We could, probably, align them to current
      // indentation level, but now we just wrap them without indentation
      // and stars.
      CommonPrefixLength = 0;
      NeedsStar = false;
    }
  } else {
    for (size_t i = 1; i < Lines.size(); ++i) {
      size_t FirstNonWhitespace = Lines[i].find_first_not_of(" ");
      if (FirstNonWhitespace != StringRef::npos) {
        NeedsStar = NeedsStar && (Lines[i][FirstNonWhitespace] == '*');
        CommonPrefixLength =
            std::min<unsigned>(CommonPrefixLength, FirstNonWhitespace);
      }
    }
  }
  if (CommonPrefixLength == UINT_MAX)
    CommonPrefixLength = 0;

  Decoration = NeedsStar ? "* " : "";

  IndentAtLineBreak =
      std::max<int>(StartColumn - OriginalStartColumn + CommonPrefixLength, 0);
}

void BreakableBlockComment::alignLines(WhitespaceManager &Whitespaces) {
  SourceLocation TokenLoc = Tok.getStartOfNonWhitespace();
  int IndentDelta = (StartColumn - 2) - OriginalStartColumn;
  if (IndentDelta > 0) {
    std::string WhiteSpace(IndentDelta, ' ');
    for (size_t i = 1; i < Lines.size(); ++i) {
      Whitespaces.addReplacement(
          TokenLoc.getLocWithOffset(Lines[i].data() - TokenText.data()), 0,
          WhiteSpace);
    }
  } else if (IndentDelta < 0) {
    std::string WhiteSpace(-IndentDelta, ' ');
    // Check that the line is indented enough.
    for (size_t i = 1; i < Lines.size(); ++i) {
      if (!Lines[i].startswith(WhiteSpace))
        return;
    }
    for (size_t i = 1; i < Lines.size(); ++i) {
      Whitespaces.addReplacement(
          TokenLoc.getLocWithOffset(Lines[i].data() - TokenText.data()),
          -IndentDelta, "");
    }
  }

  for (unsigned i = 1; i < Lines.size(); ++i)
    Lines[i] = Lines[i].substr(CommonPrefixLength + Decoration.size());
}

void BreakableBlockComment::trimLine(unsigned LineIndex, unsigned TailOffset,
                                     unsigned InPPDirective,
                                     WhitespaceManager &Whitespaces) {
  if (LineIndex == Lines.size() - 1)
    return;
  StringRef Text = Lines[LineIndex].substr(TailOffset);

  // FIXME: The algorithm for trimming a line should naturally yield a
  // non-change if there is nothing to trim; removing this line breaks the
  // algorithm; investigate the root cause, and make sure to either document
  // why exactly this is needed for remove it.
  if (!Text.endswith(" ") && !InPPDirective)
    return;

  StringRef TrimmedLine = Text.rtrim();
  unsigned BreakOffset = TrimmedLine.end() - TokenText.data();
  unsigned CharsToRemove = Text.size() - TrimmedLine.size() + 1;
  // FIXME: It seems like we're misusing the call to breakToken to remove
  // whitespace instead of breaking a token. We should make this an explicit
  // call option to the WhitespaceManager, or handle trimming and alignment
  // of comments completely within in the WhitespaceManger. Passing '0' here
  // and relying on this not breaking assumptions of the WhitespaceManager seems
  // like a bad idea.
  Whitespaces.breakToken(Tok, BreakOffset, CharsToRemove, "", "", InPPDirective,
                         0);
}

BreakableLineComment::BreakableLineComment(const SourceManager &SourceMgr,
                                           const AnnotatedToken &Token,
                                           unsigned StartColumn)
    : BreakableComment(SourceMgr, Token.FormatTok, StartColumn) {
  assert(TokenText.startswith("//"));
  Decoration = getLineCommentPrefix(TokenText);
  Lines.push_back(TokenText.substr(Decoration.size()));
  IndentAtLineBreak = StartColumn;
  this->StartColumn += Decoration.size(); // Start column of the contents.
}

StringRef BreakableLineComment::getLineCommentPrefix(StringRef Comment) {
  const char *KnownPrefixes[] = { "/// ", "///", "// ", "//" };
  for (size_t i = 0; i < llvm::array_lengthof(KnownPrefixes); ++i)
    if (Comment.startswith(KnownPrefixes[i]))
      return KnownPrefixes[i];
  return "";
}

} // namespace format
} // namespace clang
