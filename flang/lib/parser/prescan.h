// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

namespace Fortran::parser {

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
  void Say(MessageFixedText, ProvenanceRange);
  void Say(MessageFormattedText &&, ProvenanceRange);

private:
  struct LineClassification {
    enum class Kind {
      Comment,
      ConditionalCompilationDirective,
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
    slashInCurrentLine_ = false;
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

  ProvenanceRange GetProvenanceRange(
      const char *first, const char *afterLast) const {
    std::size_t bytes = afterLast - first;
    return {startProvenance_ + (first - start_), bytes};
  }

  void EmitChar(TokenSequence &tokens, char ch) {
    tokens.PutNextTokenChar(ch, GetCurrentProvenance());
  }

  void EmitInsertedChar(TokenSequence &tokens, char ch) {
    Provenance provenance{cooked_.allSources().CompilerInsertionProvenance(ch)};
    tokens.PutNextTokenChar(ch, provenance);
  }

  char EmitCharAndAdvance(TokenSequence &tokens, char ch) {
    EmitChar(tokens, ch);
    NextChar();
    return *at_;
  }

  bool InCompilerDirective() const { return directiveSentinel_ != nullptr; }
  bool InFixedFormSource() const {
    return inFixedForm_ && !inPreprocessorDirective_ && !InCompilerDirective();
  }

  void LabelField(TokenSequence &);
  void SkipToEndOfLine();
  void NextChar();
  void SkipSpaces();
  bool NextToken(TokenSequence &);
  bool ExponentAndKind(TokenSequence &);
  void QuotedCharacterLiteral(TokenSequence &);
  void Hollerith(TokenSequence &, int);
  bool PadOutCharacterLiteral(TokenSequence &);
  bool SkipCommentLine();
  bool IsFixedFormCommentLine(const char *) const;
  bool IsFreeFormComment(const char *) const;
  std::optional<std::size_t> IsIncludeLine(const char *) const;
  void FortranInclude(const char *quote);
  const char *IsPreprocessorDirectiveLine(const char *) const;
  const char *FixedFormContinuationLine(bool mightNeedSpace);
  const char *FreeFormContinuationLine(bool ampersand);
  bool FixedFormContinuation(bool mightNeedSpace);
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
  bool inFixedForm_{false};
  int fixedFormColumnLimit_{72};
  Encoding encoding_{Encoding::UTF8};
  bool enableOldDebugLines_{false};
  bool enableBackslashEscapesInCharLiterals_{true};
  bool warnOnNonstandardUsage_{false};
  int delimiterNesting_{0};
  int prescannerNesting_{0};

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
  bool slashInCurrentLine_{false};
  bool preventHollerith_{false};
  bool inCharLiteral_{false};
  bool inPreprocessorDirective_{false};

  // In some edge cases of compiler directive continuation lines, it
  // is necessary to treat the line break as a space character by
  // setting this flag, which is cleared by EmitChar().
  bool insertASpace_{false};

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

}  // namespace Fortran::parser
#endif  // FORTRAN_PARSER_PRESCAN_H_
