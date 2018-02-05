#ifndef FORTRAN_PRESCAN_H_
#define FORTRAN_PRESCAN_H_

// Defines a fast Fortran source prescanning phase that implements some
// character-level features of the language that can be inefficient to
// support directly in a backtracking parser.  This phase handles Fortran
// line continuation, comment removal, card image margins, padding out
// fixed form character literals on truncated card images, and drives the
// Fortran source preprocessor.
//
// It is possible to run the Fortran parser without running this prescan
// phase, using only the parsers defined in cooked-chars.h, so long as
// preprocessing and INCLUDE lines need not be handled.

#include "char-buffer.h"
#include "message.h"
#include "position.h"
#include "preprocessor.h"
#include "source.h"
#include <optional>

namespace Fortran {

class Prescanner {
public:
  explicit Prescanner(Messages &messages)
    : messages_{messages}, preprocessor_{*this} {}

  Messages &messages() const { return messages_; }
  const SourceFile &sourceFile() const { return *sourceFile_; }
  Position position() const { return atPosition_; }
  bool anyFatalErrors() const { return anyFatalErrors_; }

  Prescanner &set_fixedForm(bool yes) {
    inFixedForm_ = yes;
    return *this;
  }
  Prescanner &set_enableOldDebugLines(bool yes) {
    enableOldDebugLines_ = yes;
    return *this;
  }
  Prescanner &set_enableBackslashEscapesInCharLiterals(bool yes) {
    enableBackslashEscapesInCharLiterals_ = yes;
    return *this;
  }
  Prescanner &set_fixedFormColumnLimit(int limit) {
    fixedFormColumnLimit_ = limit;
    return *this;
  }

  CharBuffer Prescan(const SourceFile &source);
  std::optional<TokenSequence> NextTokenizedLine();

private:
  void BeginSourceLine(const char *at) {
    at_ = at;
    atPosition_ = lineStartPosition_;
    tabInCurrentLine_ = false;
    preventHollerith_ = false;
    delimiterNesting_ = 0;
  }

  void BeginSourceLineAndAdvance() {
    BeginSourceLine(lineStart_);
    NextLine();
  }

  char EmitCharAndAdvance(TokenSequence *tokens, char ch) {
    tokens->AddChar(ch);
    NextChar();
    return *at_;
  }

  void NextLine();
  void LabelField(TokenSequence *);
  void NextChar();
  void SkipSpaces();
  bool NextToken(TokenSequence *);
  bool ExponentAndKind(TokenSequence *);
  void QuotedCharacterLiteral(TokenSequence *);
  bool PadOutCharacterLiteral();
  bool CommentLines();
  bool CommentLinesAndPreprocessorDirectives();
  bool IsFixedFormCommentLine(const char *);
  bool IsFreeFormComment(const char *);
  bool IsPreprocessorDirectiveLine(const char *);
  const char *FixedFormContinuationLine();
  bool FixedFormContinuation();
  bool FreeFormContinuation();
  void PayNewlineDebt(CharBuffer *);

  Messages &messages_;
  bool anyFatalErrors_{false};
  const char *lineStart_{nullptr};  // next line to process; <= limit_
  const char *at_{nullptr};  // next character to process; < lineStart_
  int column_{1};  // card image column position of next character
  const char *limit_{nullptr};  // first address after end of source
  int newlineDebt_{0};  // newline characters consumed but not yet emitted
  const SourceFile *sourceFile_{nullptr};
  Position atPosition_, lineStartPosition_;
  bool inCharLiteral_{false};
  bool inPreprocessorDirective_{false};
  bool inFixedForm_{true};
  int fixedFormColumnLimit_{72};
  bool tabInCurrentLine_{false};
  bool preventHollerith_{false};
  bool enableOldDebugLines_{false};
  bool enableBackslashEscapesInCharLiterals_{true};
  int delimiterNesting_{0};
  Preprocessor preprocessor_;
};
}  // namespace Fortran
#endif  // FORTRAN_PRESCAN_H_
