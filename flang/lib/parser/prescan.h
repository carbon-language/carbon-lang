#ifndef FORTRAN_PARSER_PRESCAN_H_
#define FORTRAN_PARSER_PRESCAN_H_

// Defines a fast Fortran source prescanning phase that implements some
// character-level features of the language that can be inefficient to
// support directly in a backtracking parser.  This phase handles Fortran
// line continuation, comment removal, card image margins, padding out
// fixed form character literals on truncated card images, file
// inclusion, and driving the Fortran source preprocessor.

#include "provenance.h"
#include "token-sequence.h"
#include <optional>
#include <string>

namespace Fortran {
namespace parser {

class Messages;
class Preprocessor;

class Prescanner {
public:
  Prescanner(Messages *, CookedSource *, Preprocessor *);
  Prescanner(const Prescanner &);

  Messages *messages() const { return messages_; }

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

  bool Prescan(ProvenanceRange);

  // Callbacks for use by Preprocessor.
  std::optional<TokenSequence> NextTokenizedLine();
  Provenance GetCurrentProvenance() const { return GetProvenance(at_); }
  void Complain(const std::string &message);

private:
  void BeginSourceLine(const char *at) {
    at_ = at;
    column_ = 1;
    tabInCurrentLine_ = false;
    preventHollerith_ = false;
    delimiterNesting_ = 0;
  }

  void BeginSourceLineAndAdvance() {
    BeginSourceLine(lineStart_);
    NextLine();
  }

  Provenance GetProvenance(const char *sourceChar) const {
    return startProvenance_ + (sourceChar - start_);
  }

  void EmitChar(TokenSequence *tokens, char ch) {
    tokens->PutNextTokenChar(ch, GetCurrentProvenance());
  }

  char EmitCharAndAdvance(TokenSequence *tokens, char ch) {
    EmitChar(tokens, ch);
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
  bool IncludeLine(const char *);
  bool IsPreprocessorDirectiveLine(const char *);
  const char *FixedFormContinuationLine();
  bool FixedFormContinuation();
  bool FreeFormContinuation();
  void PayNewlineDebt(Provenance);

  Messages *messages_;
  CookedSource *cooked_;
  Preprocessor *preprocessor_;

  Provenance startProvenance_;
  const char *start_{nullptr};  // beginning of current source file content
  const char *limit_{nullptr};  // first address after end of current source
  const char *at_{nullptr};  // next character to process; < lineStart_
  int column_{1};  // card image column position of next character
  const char *lineStart_{nullptr};  // next line to process; <= limit_
  bool tabInCurrentLine_{false};
  bool preventHollerith_{false};

  bool anyFatalErrors_{false};
  int newlineDebt_{0};  // newline characters consumed but not yet emitted
  bool inCharLiteral_{false};
  bool inPreprocessorDirective_{false};
  bool inFixedForm_{false};
  int fixedFormColumnLimit_{72};
  bool enableOldDebugLines_{false};
  bool enableBackslashEscapesInCharLiterals_{true};
  int delimiterNesting_{0};
  Provenance spaceProvenance_{
      cooked_->allSources()->CompilerInsertionProvenance(' ')};
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_PRESCAN_H_
