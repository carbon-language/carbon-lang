#ifndef FORTRAN_PARSER_PRESCAN_H_
#define FORTRAN_PARSER_PRESCAN_H_

// Defines a fast Fortran source prescanning phase that implements some
// character-level features of the language that can be inefficient to
// support directly in a backtracking parser.  This phase handles Fortran
// line continuation, comment removal, card image margins, padding out
// fixed form character literals on truncated card images, file
// inclusion, and driving the Fortran source preprocessor.

#include "characters.h"
#include "message.h"
#include "provenance.h"
#include "token-sequence.h"
#include <bitset>
#include <optional>
#include <string>
#include <unordered_set>

namespace Fortran {
namespace parser {

class Messages;
class Preprocessor;

class Prescanner {
public:
  Prescanner(Messages &, CookedSource &, Preprocessor &);
  Prescanner(const Prescanner &);

  Messages &messages() const { return messages_; }

  Prescanner &set_fixedForm(bool yes) {
    inFixedForm_ = yes;
    return *this;
  }
  Prescanner &set_encoding(Encoding code) {
    encoding_ = code;
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
  Prescanner &set_warnOnNonstandardUsage(bool yes) {
    warnOnNonstandardUsage_ = yes;
    return *this;
  }

  Prescanner &AddCompilerDirectiveSentinel(const std::string &);

  void Prescan(ProvenanceRange);
  void Statement();
  void NextLine();

  // Callbacks for use by Preprocessor.
  bool IsAtEnd() const { return lineStart_ >= limit_; }
  bool IsNextLinePreprocessorDirective() const;
  TokenSequence TokenizePreprocessorDirective();
  Provenance GetCurrentProvenance() const { return GetProvenance(at_); }

  void Say(Message &&);
  void Say(MessageFixedText, Provenance);
  void Say(MessageFormattedText &&, Provenance);

private:
  struct LineClassification {
    enum class Kind {
      Comment,
      PreprocessorDirective,
      Include,
      CompilerDirective,
      Source
    };
    LineClassification(Kind k, std::size_t po = 0, const char *s = nullptr)
      : kind{k}, payloadOffset{po}, sentinel{s} {}
    LineClassification(LineClassification &&) = default;
    Kind kind;
    std::size_t payloadOffset;  // byte offset of content
    const char *sentinel;  // if it's a compiler directive
  };

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

  void EmitInsertedChar(TokenSequence *tokens, char ch) {
    Provenance provenance{cooked_.allSources().CompilerInsertionProvenance(ch)};
    tokens->PutNextTokenChar(ch, provenance);
  }

  char EmitCharAndAdvance(TokenSequence *tokens, char ch) {
    EmitChar(tokens, ch);
    NextChar();
    return *at_;
  }

  void LabelField(TokenSequence *);
  void NextChar();
  void SkipSpaces();
  bool NextToken(TokenSequence *);
  bool ExponentAndKind(TokenSequence *);
  void QuotedCharacterLiteral(TokenSequence *);
  void Hollerith(TokenSequence *, int);
  bool PadOutCharacterLiteral(TokenSequence *);
  void SkipCommentLines();
  bool IsFixedFormCommentLine(const char *) const;
  bool IsFreeFormComment(const char *) const;
  std::optional<std::size_t> IsIncludeLine(const char *) const;
  bool FortranInclude(const char *quote);
  bool IsPreprocessorDirectiveLine(const char *) const;
  const char *FixedFormContinuationLine();
  bool FixedFormContinuation();
  bool FreeFormContinuation();
  std::optional<LineClassification> IsFixedFormCompilerDirectiveLine(
      const char *) const;
  std::optional<LineClassification> IsFreeFormCompilerDirectiveLine(
      const char *) const;
  const char *IsCompilerDirectiveSentinel(const char *) const;
  LineClassification ClassifyLine(const char *) const;
  void SourceFormChange(std::string &&);

  Messages &messages_;
  CookedSource &cooked_;
  Preprocessor &preprocessor_;
  bool anyFatalErrors_{false};
  bool inFixedForm_{false};
  int fixedFormColumnLimit_{72};
  Encoding encoding_{Encoding::UTF8};
  bool enableOldDebugLines_{false};
  bool enableBackslashEscapesInCharLiterals_{true};
  bool warnOnNonstandardUsage_{false};
  int delimiterNesting_{0};

  Provenance startProvenance_;
  const char *start_{nullptr};  // beginning of current source file content
  const char *limit_{nullptr};  // first address after end of current source
  const char *lineStart_{nullptr};  // next line to process; <= limit_
  const char *directiveSentinel_{nullptr};  // current compiler directive

  // This data members are state for processing the source line containing
  // "at_", which goes to up to the newline character before "lineStart_".
  const char *at_{nullptr};  // next character to process; < lineStart_
  int column_{1};  // card image column position of next character
  bool tabInCurrentLine_{false};
  bool preventHollerith_{false};
  bool inCharLiteral_{false};
  bool inPreprocessorDirective_{false};

  const Provenance spaceProvenance_{
      cooked_.allSources().CompilerInsertionProvenance(' ')};
  const Provenance backslashProvenance_{
      cooked_.allSources().CompilerInsertionProvenance('\\')};
  const ProvenanceRange sixSpaceProvenance_{
      cooked_.allSources().AddCompilerInsertion("      "s)};

  // To avoid probing the set of active compiler directive sentinel strings
  // on every comment line, they're checked first with a cheap Bloom filter.
  static const int prime1{1019}, prime2{1021};
  std::bitset<prime2> compilerDirectiveBloomFilter_;  // 128 bytes
  std::unordered_set<std::string> compilerDirectiveSentinels_;
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_PRESCAN_H_
