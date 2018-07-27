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

#include "prescan.h"
#include "characters.h"
#include "message.h"
#include "preprocessor.h"
#include "source.h"
#include "token-sequence.h"
#include "../common/idioms.h"
#include <cstddef>
#include <cstring>
#include <sstream>
#include <utility>
#include <vector>

namespace Fortran::parser {

static constexpr int maxPrescannerNesting{100};

Prescanner::Prescanner(Messages &messages, CookedSource &cooked,
    Preprocessor &preprocessor, LanguageFeatureControl lfc)
  : messages_{messages}, cooked_{cooked},
    preprocessor_{preprocessor}, features_{lfc} {}

Prescanner::Prescanner(const Prescanner &that)
  : messages_{that.messages_}, cooked_{that.cooked_},
    preprocessor_{that.preprocessor_}, features_{that.features_},
    inFixedForm_{that.inFixedForm_},
    fixedFormColumnLimit_{that.fixedFormColumnLimit_},
    encoding_{that.encoding_}, prescannerNesting_{that.prescannerNesting_ + 1},
    compilerDirectiveBloomFilter_{that.compilerDirectiveBloomFilter_},
    compilerDirectiveSentinels_{that.compilerDirectiveSentinels_} {}

static inline constexpr bool IsFixedFormCommentChar(char ch) {
  return ch == '!' || ch == '*' || ch == 'C' || ch == 'c';
}

static void NormalizeCompilerDirectiveCommentMarker(TokenSequence &dir) {
  char *p{dir.GetMutableCharData()};
  char *limit{p + dir.SizeInChars()};
  for (; p < limit; ++p) {
    if (*p != ' ') {
      CHECK(IsFixedFormCommentChar(*p));
      *p = '!';
      return;
    }
  }
  CHECK(!"compiler directive all blank");
}

void Prescanner::Prescan(ProvenanceRange range) {
  AllSources &allSources{cooked_.allSources()};
  startProvenance_ = range.start();
  std::size_t offset{0};
  const SourceFile *source{allSources.GetSourceFile(startProvenance_, &offset)};
  CHECK(source != nullptr);
  start_ = source->content() + offset;
  limit_ = start_ + range.size();
  lineStart_ = start_;
  const bool beganInFixedForm{inFixedForm_};
  if (prescannerNesting_ > maxPrescannerNesting) {
    Say("too many nested INCLUDE/#include files, possibly circular"_err_en_US,
        GetProvenance(start_));
    return;
  }
  while (lineStart_ < limit_) {
    Statement();
  }
  if (inFixedForm_ != beganInFixedForm) {
    std::string dir{"!dir$ "};
    if (beganInFixedForm) {
      dir += "fixed";
    } else {
      dir += "free";
    }
    dir += '\n';
    TokenSequence tokens{dir, allSources.AddCompilerInsertion(dir).start()};
    tokens.Emit(cooked_);
  }
}

void Prescanner::Statement() {
  TokenSequence tokens;
  LineClassification line{ClassifyLine(lineStart_)};
  switch (line.kind) {
  case LineClassification::Kind::Comment: NextLine(); return;
  case LineClassification::Kind::Include:
    FortranInclude(lineStart_ + line.payloadOffset);
    NextLine();
    return;
  case LineClassification::Kind::ConditionalCompilationDirective:
  case LineClassification::Kind::PreprocessorDirective:
    preprocessor_.Directive(TokenizePreprocessorDirective(), this);
    return;
  case LineClassification::Kind::CompilerDirective:
    directiveSentinel_ = line.sentinel;
    CHECK(InCompilerDirective());
    BeginSourceLineAndAdvance();
    if (inFixedForm_) {
      CHECK(IsFixedFormCommentChar(*at_));
    } else {
      while (*at_ == ' ' || *at_ == '\t') {
        ++at_, ++column_;
      }
      CHECK(*at_ == '!');
    }
    if (directiveSentinel_[0] == '$' && directiveSentinel_[1] == '\0') {
      // OpenMP conditional compilation line.  Remove the sentinel and then
      // treat the line as if it were normal source.
      at_ += 2, column_ += 2;
      if (inFixedForm_) {
        LabelField(tokens);
      } else {
        SkipSpaces();
      }
    } else {
      // Compiler directive.  Emit normalized sentinel.
      EmitChar(tokens, '!');
      ++at_, ++column_;
      for (const char *sp{directiveSentinel_}; *sp != '\0';
           ++sp, ++at_, ++column_) {
        EmitChar(tokens, *sp);
      }
      if (*at_ == ' ') {
        EmitChar(tokens, ' ');
        ++at_, ++column_;
      }
      tokens.CloseToken();
    }
    break;
  case LineClassification::Kind::Source:
    BeginSourceLineAndAdvance();
    if (inFixedForm_) {
      LabelField(tokens);
    } else {
      SkipSpaces();
    }
    break;
  }

  while (NextToken(tokens)) {
  }

  Provenance newlineProvenance{GetCurrentProvenance()};
  if (std::optional<TokenSequence> preprocessed{
          preprocessor_.MacroReplacement(tokens, *this)}) {
    // Reprocess the preprocessed line.
    preprocessed->PutNextTokenChar('\n', newlineProvenance);
    preprocessed->CloseToken();
    const char *ppd{preprocessed->ToCharBlock().begin()};
    LineClassification ppl{ClassifyLine(ppd)};
    switch (ppl.kind) {
    case LineClassification::Kind::Comment: break;
    case LineClassification::Kind::Include:
      FortranInclude(ppd + ppl.payloadOffset);
      break;
    case LineClassification::Kind::ConditionalCompilationDirective:
    case LineClassification::Kind::PreprocessorDirective:
      Say("preprocessed line resembles a preprocessor directive"_en_US,
          preprocessed->GetProvenanceRange());
      preprocessed->ToLowerCase().Emit(cooked_);
      break;
    case LineClassification::Kind::CompilerDirective:
      if (preprocessed->HasRedundantBlanks()) {
        preprocessed->RemoveRedundantBlanks();
      }
      NormalizeCompilerDirectiveCommentMarker(*preprocessed);
      preprocessed->ToLowerCase();
      SourceFormChange(preprocessed->ToString());
      preprocessed->Emit(cooked_);
      break;
    case LineClassification::Kind::Source:
      if (inFixedForm_) {
        if (preprocessed->HasBlanks(/*after column*/ 6)) {
          preprocessed->RemoveBlanks(/*after column*/ 6);
        }
      } else {
        if (preprocessed->HasRedundantBlanks()) {
          preprocessed->RemoveRedundantBlanks();
        }
      }
      preprocessed->ToLowerCase().Emit(cooked_);
      break;
    }
  } else {
    tokens.ToLowerCase();
    if (line.kind == LineClassification::Kind::CompilerDirective) {
      SourceFormChange(tokens.ToString());
    }
    tokens.Emit(cooked_);
    cooked_.Put('\n', newlineProvenance);
  }
  directiveSentinel_ = nullptr;
}

TokenSequence Prescanner::TokenizePreprocessorDirective() {
  CHECK(lineStart_ < limit_ && !inPreprocessorDirective_);
  auto saveAt{at_};
  inPreprocessorDirective_ = true;
  BeginSourceLineAndAdvance();
  TokenSequence tokens;
  while (NextToken(tokens)) {
  }
  inPreprocessorDirective_ = false;
  at_ = saveAt;
  return tokens;
}

void Prescanner::Say(Message &&message) {
  std::optional<ProvenanceRange> range{message.GetProvenanceRange(cooked_)};
  CHECK(!range.has_value() || cooked_.IsValid(*range));
  messages_.Put(std::move(message));
}

void Prescanner::Say(MessageFixedText text, ProvenanceRange r) {
  CHECK(cooked_.IsValid(r));
  messages_.Put({r, text});
}

void Prescanner::Say(MessageFormattedText &&text, ProvenanceRange r) {
  CHECK(cooked_.IsValid(r));
  messages_.Put({r, std::move(text)});
}

void Prescanner::NextLine() {
  void *vstart{static_cast<void *>(const_cast<char *>(lineStart_))};
  void *v{std::memchr(vstart, '\n', limit_ - lineStart_)};
  if (v == nullptr) {
    lineStart_ = limit_;
  } else {
    const char *nl{const_cast<const char *>(static_cast<char *>(v))};
    lineStart_ = nl + 1;
  }
}

void Prescanner::LabelField(TokenSequence &token, int outCol) {
  for (; *at_ != '\n' && column_ <= 6; ++at_) {
    if (*at_ == '\t') {
      ++at_;
      column_ = 7;
      break;
    }
    if (*at_ != ' ' &&
        !(*at_ == '0' && column_ == 6)) {  // '0' in column 6 becomes space
      EmitChar(token, *at_);
      ++outCol;
    }
    ++column_;
  }
  if (outCol > 1) {
    token.CloseToken();
  }
  if (outCol < 7) {
    if (outCol == 1) {
      token.Put("      ", 6, sixSpaceProvenance_.start());
    } else {
      for (; outCol < 7; ++outCol) {
        token.PutNextTokenChar(' ', spaceProvenance_);
      }
      token.CloseToken();
    }
  }
}

void Prescanner::SkipToEndOfLine() {
  while (*at_ != '\n') {
    ++at_, ++column_;
  }
}

void Prescanner::NextChar() {
  CHECK(*at_ != '\n');
  ++at_, ++column_;
  if (inPreprocessorDirective_) {
    while (*at_ == '/' && at_[1] == '*') {
      char star{' '}, slash{' '};
      at_ += 2;
      column_ += 2;
      while ((*at_ != '\n' || slash == '\\') && (star != '*' || slash != '/')) {
        star = slash;
        slash = *at_++;
        ++column_;
      }
    }
    while (*at_ == '\\' && at_ + 2 < limit_ && at_[1] == '\n') {
      BeginSourceLineAndAdvance();
    }
  } else {
    bool rightMarginClip{
        inFixedForm_ && column_ > fixedFormColumnLimit_ && !tabInCurrentLine_};
    bool skipping{rightMarginClip || (*at_ == '!' && !inCharLiteral_)};
    if (skipping) {
      SkipToEndOfLine();
    }
    while (*at_ == '\n' || *at_ == '&') {
      bool mightNeedSpace{*at_ == '\n' && !skipping};
      if (inFixedForm_) {
        if (!FixedFormContinuation(mightNeedSpace)) {
          return;
        }
      } else if (!FreeFormContinuation()) {
        return;
      }
    }
    if (*at_ == '\t') {
      tabInCurrentLine_ = true;
    }
  }
}

void Prescanner::SkipSpaces() {
  while (*at_ == ' ' || *at_ == '\t') {
    NextChar();
  }
  insertASpace_ = false;
}

bool Prescanner::NextToken(TokenSequence &tokens) {
  CHECK(at_ >= start_ && at_ < limit_);
  if (InFixedFormSource()) {
    SkipSpaces();
  } else if (*at_ == ' ' || *at_ == '\t') {
    // Compress white space into a single space character.
    // Discard white space at the end of a line.
    const auto theSpace{at_};
    NextChar();
    SkipSpaces();
    if (*at_ != '\n') {
      tokens.PutNextTokenChar(' ', GetProvenance(theSpace));
      tokens.CloseToken();
      return true;
    }
  }
  if (insertASpace_) {
    tokens.PutNextTokenChar(' ', spaceProvenance_);
    insertASpace_ = false;
  }
  if (*at_ == '\n') {
    return false;
  }
  if (*at_ == '\'' || *at_ == '"') {
    QuotedCharacterLiteral(tokens);
    preventHollerith_ = false;
  } else if (IsDecimalDigit(*at_)) {
    int n{0}, digits{0};
    static constexpr int maxHollerith{256 /*lines*/ * (132 - 6 /*columns*/)};
    do {
      if (n < maxHollerith) {
        n = 10 * n + DecimalDigitValue(*at_);
      }
      EmitCharAndAdvance(tokens, *at_);
      ++digits;
      if (InFixedFormSource()) {
        SkipSpaces();
      }
    } while (IsDecimalDigit(*at_));
    if ((*at_ == 'h' || *at_ == 'H') && n > 0 && n < maxHollerith &&
        !preventHollerith_) {
      Hollerith(tokens, n);
    } else if (*at_ == '.') {
      while (IsDecimalDigit(EmitCharAndAdvance(tokens, *at_))) {
      }
      ExponentAndKind(tokens);
    } else if (ExponentAndKind(tokens)) {
    } else if (digits == 1 && n == 0 && (*at_ == 'x' || *at_ == 'X') &&
        inPreprocessorDirective_) {
      do {
        EmitCharAndAdvance(tokens, *at_);
      } while (IsHexadecimalDigit(*at_));
    } else if (IsLetter(*at_)) {
      // Handles FORMAT(3I9HHOLLERITH) by skipping over the first I so that
      // we don't misrecognize I9HOLLERITH as an identifier in the next case.
      EmitCharAndAdvance(tokens, *at_);
    }
    preventHollerith_ = false;
  } else if (*at_ == '.') {
    char nch{EmitCharAndAdvance(tokens, '.')};
    if (!inPreprocessorDirective_ && IsDecimalDigit(nch)) {
      while (IsDecimalDigit(EmitCharAndAdvance(tokens, *at_))) {
      }
      ExponentAndKind(tokens);
    } else if (nch == '.' && EmitCharAndAdvance(tokens, '.') == '.') {
      EmitCharAndAdvance(tokens, '.');  // variadic macro definition ellipsis
    }
    preventHollerith_ = false;
  } else if (IsLegalInIdentifier(*at_)) {
    while (IsLegalInIdentifier(EmitCharAndAdvance(tokens, *at_))) {
    }
    if (*at_ == '\'' || *at_ == '"') {
      QuotedCharacterLiteral(tokens);
    }
    preventHollerith_ = false;
  } else if (*at_ == '*') {
    if (EmitCharAndAdvance(tokens, '*') == '*') {
      EmitCharAndAdvance(tokens, '*');
    } else {
      // Subtle ambiguity:
      //  CHARACTER*2H     declares H because *2 is a kind specifier
      //  DATAC/N*2H  /    is repeated Hollerith
      preventHollerith_ = !slashInCurrentLine_;
    }
  } else {
    char ch{*at_};
    if (ch == '(' || ch == '[') {
      ++delimiterNesting_;
    } else if ((ch == ')' || ch == ']') && delimiterNesting_ > 0) {
      --delimiterNesting_;
    }
    char nch{EmitCharAndAdvance(tokens, ch)};
    preventHollerith_ = false;
    if ((nch == '=' &&
            (ch == '<' || ch == '>' || ch == '/' || ch == '=' || ch == '!')) ||
        (ch == nch &&
            (ch == '/' || ch == ':' || ch == '*' || ch == '#' || ch == '&' ||
                ch == '|' || ch == '<' || ch == '>')) ||
        (ch == '=' && nch == '>')) {
      // token comprises two characters
      EmitCharAndAdvance(tokens, nch);
    } else if (ch == '/') {
      slashInCurrentLine_ = true;
    }
  }
  tokens.CloseToken();
  return true;
}

bool Prescanner::ExponentAndKind(TokenSequence &tokens) {
  char ed = ToLowerCaseLetter(*at_);
  if (ed != 'e' && ed != 'd') {
    return false;
  }
  EmitCharAndAdvance(tokens, ed);
  if (*at_ == '+' || *at_ == '-') {
    EmitCharAndAdvance(tokens, *at_);
  }
  while (IsDecimalDigit(*at_)) {
    EmitCharAndAdvance(tokens, *at_);
  }
  if (*at_ == '_') {
    while (IsLegalInIdentifier(EmitCharAndAdvance(tokens, *at_))) {
    }
  }
  return true;
}

void Prescanner::QuotedCharacterLiteral(TokenSequence &tokens) {
  const char *start{at_}, quote{*start}, *end{at_ + 1};
  inCharLiteral_ = true;
  const auto emit{[&](char ch) { EmitChar(tokens, ch); }};
  const auto insert{[&](char ch) { EmitInsertedChar(tokens, ch); }};
  bool escape{false};
  bool escapesEnabled{features_.IsEnabled(LanguageFeature::BackslashEscapes)};
  while (true) {
    char ch{*at_};
    escape = !escape && ch == '\\' && escapesEnabled;
    EmitQuotedChar(ch, emit, insert, false, !escapesEnabled);
    while (PadOutCharacterLiteral(tokens)) {
    }
    if (*at_ == '\n') {
      if (!inPreprocessorDirective_) {
        Say("incomplete character literal"_err_en_US,
            GetProvenanceRange(start, end));
      }
      break;
    }
    end = at_ + 1;
    NextChar();
    if (*at_ == quote && !escape) {
      // A doubled quote mark becomes a single instance of the quote character
      // in the literal (later).  There can be spaces between the quotes in
      // fixed form source.
      EmitChar(tokens, quote);
      inCharLiteral_ = false;  // for cases like print *, '...'!comment
      NextChar();
      if (InFixedFormSource()) {
        SkipSpaces();
      }
      if (*at_ != quote) {
        break;
      }
      inCharLiteral_ = true;
    }
  }
  inCharLiteral_ = false;
}

void Prescanner::Hollerith(TokenSequence &tokens, int count) {
  inCharLiteral_ = true;
  CHECK(*at_ == 'h' || *at_ == 'H');
  EmitChar(tokens, 'H');
  const char *start{at_}, *end{at_ + 1};
  while (count-- > 0) {
    if (PadOutCharacterLiteral(tokens)) {
    } else if (*at_ == '\n') {
      Say("incomplete Hollerith literal"_err_en_US,
          GetProvenanceRange(start, end));
      break;
    } else {
      end = at_ + 1;
      NextChar();
      EmitChar(tokens, *at_);
      // Multi-byte character encodings should count as single characters.
      int bytes{1};
      if (encoding_ == Encoding::EUC_JP) {
        if (std::optional<int> chBytes{EUC_JPCharacterBytes(at_)}) {
          bytes = *chBytes;
        }
      } else if (encoding_ == Encoding::UTF8) {
        if (std::optional<int> chBytes{UTF8CharacterBytes(at_)}) {
          bytes = *chBytes;
        }
      }
      while (bytes-- > 1) {
        EmitChar(tokens, *++at_);
      }
    }
  }
  if (*at_ != '\n') {
    NextChar();
  }
  inCharLiteral_ = false;
}

// In fixed form, source card images must be processed as if they were at
// least 72 columns wide, at least in character literal contexts.
bool Prescanner::PadOutCharacterLiteral(TokenSequence &tokens) {
  while (inFixedForm_ && !tabInCurrentLine_ && at_[1] == '\n') {
    if (column_ < fixedFormColumnLimit_) {
      tokens.PutNextTokenChar(' ', spaceProvenance_);
      ++column_;
      return true;
    }
    if (!FixedFormContinuation(false /*no need to insert space*/) ||
        tabInCurrentLine_) {
      return false;
    }
    CHECK(column_ == 7);
    --at_;  // point to column 6 of continuation line
    column_ = 6;
  }
  return false;
}

bool Prescanner::IsFixedFormCommentLine(const char *start) const {
  const char *p{start};
  char ch{*p};
  if (IsFixedFormCommentChar(ch) || ch == '%' ||  // VAX %list, %eject, &c.
      ((ch == 'D' || ch == 'd') &&
          !features_.IsEnabled(LanguageFeature::OldDebugLines))) {
    return true;
  }
  bool anyTabs{false};
  while (true) {
    ch = *p;
    if (ch == ' ') {
      ++p;
    } else if (ch == '\t') {
      anyTabs = true;
      ++p;
    } else if (ch == '0' && !anyTabs && p == start + 5) {
      ++p;  // 0 in column 6 must treated as a space
    } else {
      break;
    }
  }
  if (!anyTabs && p >= start + fixedFormColumnLimit_) {
    return true;
  }
  if (*p == '!' && !inCharLiteral_ && (anyTabs || p != start + 5)) {
    return true;
  }
  return *p == '\n';
}

bool Prescanner::IsFreeFormComment(const char *p) const {
  while (*p == ' ' || *p == '\t') {
    ++p;
  }
  return *p == '!' || *p == '\n';
}

std::optional<std::size_t> Prescanner::IsIncludeLine(const char *start) const {
  const char *p{start};
  while (*p == ' ' || *p == '\t') {
    ++p;
  }
  for (char ch : "include"s) {
    if (ToLowerCaseLetter(*p++) != ch) {
      return {};
    }
  }
  while (*p == ' ' || *p == '\t') {
    ++p;
  }
  if (*p == '"' || *p == '\'') {
    return {p - start};
  }
  return {};
}

void Prescanner::FortranInclude(const char *firstQuote) {
  const char *p{firstQuote};
  while (*p != '"' && *p != '\'') {
    ++p;
  }
  char quote{*p};
  std::string path;
  for (++p; *p != '\n'; ++p) {
    if (*p == quote) {
      if (p[1] != quote) {
        break;
      }
      ++p;
    }
    path += *p;
  }
  if (*p != quote) {
    Say("malformed path name string"_err_en_US,
        GetProvenanceRange(firstQuote, p));
    return;
  }
  for (++p; *p == ' ' || *p == '\t'; ++p) {
  }
  if (*p != '\n' && *p != '!') {
    const char *garbage{p};
    for (; *p != '\n' && *p != '!'; ++p) {
    }
    Say("excess characters after path name"_en_US,
        GetProvenanceRange(garbage, p));
  }
  std::stringstream error;
  Provenance provenance{GetProvenance(lineStart_)};
  AllSources &allSources{cooked_.allSources()};
  const SourceFile *currentFile{allSources.GetSourceFile(provenance)};
  if (currentFile != nullptr) {
    allSources.PushSearchPathDirectory(DirectoryName(currentFile->path()));
  }
  const SourceFile *included{allSources.Open(path, &error)};
  if (currentFile != nullptr) {
    allSources.PopSearchPathDirectory();
  }
  if (included == nullptr) {
    Say(MessageFormattedText("INCLUDE: %s"_err_en_US, error.str().data()),
        provenance);
  } else if (included->bytes() > 0) {
    ProvenanceRange includeLineRange{
        provenance, static_cast<std::size_t>(p - lineStart_)};
    ProvenanceRange fileRange{
        allSources.AddIncludedFile(*included, includeLineRange)};
    Prescanner{*this}.Prescan(fileRange);
  }
}

const char *Prescanner::IsPreprocessorDirectiveLine(const char *start) const {
  const char *p{start};
  for (; *p == ' '; ++p) {
  }
  if (*p == '#') {
    if (inFixedForm_ && p == start + 5) {
      return nullptr;
    }
  } else {
    for (; *p == ' ' || *p == '\t'; ++p) {
    }
    if (*p != '#') {
      return nullptr;
    }
  }
  for (++p; *p == ' ' || *p == '\t'; ++p) {
  }
  return p;
}

bool Prescanner::IsNextLinePreprocessorDirective() const {
  return IsPreprocessorDirectiveLine(lineStart_) != nullptr;
}

bool Prescanner::SkipCommentLine() {
  if (lineStart_ >= limit_) {
    return false;
  }
  auto lineClass{ClassifyLine(lineStart_)};
  if (lineClass.kind == LineClassification::Kind::Comment) {
    NextLine();
    return true;
  } else if (!inPreprocessorDirective_ &&
      lineClass.kind ==
          LineClassification::Kind::ConditionalCompilationDirective) {
    // Allow conditional compilation directives (e.g., #ifdef) to affect
    // continuation lines.
    preprocessor_.Directive(TokenizePreprocessorDirective(), this);
    return true;
  } else {
    return false;
  }
}

const char *Prescanner::FixedFormContinuationLine(bool mightNeedSpace) {
  if (lineStart_ >= limit_) {
    return nullptr;
  }
  tabInCurrentLine_ = false;
  char col1{*lineStart_};
  if (InCompilerDirective()) {
    // Must be a continued compiler directive.
    if (!IsFixedFormCommentChar(col1)) {
      return nullptr;
    }
    int j{1};
    for (; j < 5; ++j) {
      char ch{directiveSentinel_[j - 1]};
      if (ch == '\0') {
        break;
      }
      if (ch != ToLowerCaseLetter(lineStart_[j])) {
        return nullptr;
      }
    }
    for (; j < 5; ++j) {
      if (lineStart_[j] != ' ') {
        return nullptr;
      }
    }
    char col6{lineStart_[5]};
    if (col6 != '\n' && col6 != '\t' && col6 != ' ' && col6 != '0') {
      if (lineStart_[6] != ' ' && mightNeedSpace) {
        insertASpace_ = true;
      }
      return lineStart_ + 6;
    }
    return nullptr;
  } else {
    // Normal case: not in a compiler directive.
    if (col1 == '&' &&
        features_.IsEnabled(
            LanguageFeature::FixedFormContinuationWithColumn1Ampersand)) {
      // Extension: '&' as continuation marker
      if (features_.ShouldWarn(
              LanguageFeature::FixedFormContinuationWithColumn1Ampersand)) {
        Say("nonstandard usage"_en_US, GetProvenance(lineStart_));
      }
      return lineStart_ + 1;
    }
    if (col1 == '\t' && lineStart_[1] >= '1' && lineStart_[1] <= '9') {
      tabInCurrentLine_ = true;
      return lineStart_ + 2;  // VAX extension
    }
    if (col1 == ' ' && lineStart_[1] == ' ' && lineStart_[2] == ' ' &&
        lineStart_[3] == ' ' && lineStart_[4] == ' ') {
      char col6{lineStart_[5]};
      if (col6 != '\n' && col6 != '\t' && col6 != ' ' && col6 != '0') {
        return lineStart_ + 6;
      }
    }
    if (delimiterNesting_ > 0) {
      if (!IsFixedFormCommentChar(col1)) {
        return lineStart_;
      }
    }
  }
  return nullptr;  // not a continuation line
}

const char *Prescanner::FreeFormContinuationLine(bool ampersand) {
  const char *p{lineStart_};
  if (p >= limit_) {
    return nullptr;
  }
  for (; *p == ' ' || *p == '\t'; ++p) {
  }
  if (InCompilerDirective()) {
    if (*p++ != '!') {
      return nullptr;
    }
    for (const char *s{directiveSentinel_}; *s != '\0'; ++p, ++s) {
      if (*s != ToLowerCaseLetter(*p)) {
        return nullptr;
      }
    }
    for (; *p == ' ' || *p == '\t'; ++p) {
    }
    if (*p == '&') {
      if (!ampersand) {
        insertASpace_ = true;
      }
      return p + 1;
    } else if (ampersand) {
      return p;
    } else {
      return nullptr;
    }
  } else {
    if (*p == '&') {
      return p + 1;
    } else if (*p == '!' || *p == '\n') {
      return nullptr;
    } else if (ampersand || delimiterNesting_ > 0) {
      if (p > lineStart_) {
        --p;
      } else {
        insertASpace_ = true;
      }
      return p;
    } else {
      return nullptr;
    }
  }
}

bool Prescanner::FixedFormContinuation(bool mightNeedSpace) {
  // N.B. We accept '&' as a continuation indicator in fixed form, too,
  // but not in a character literal.
  if (*at_ == '&' && inCharLiteral_) {
    return false;
  }
  do {
    if (const char *cont{FixedFormContinuationLine(mightNeedSpace)}) {
      BeginSourceLine(cont);
      column_ = 7;
      NextLine();
      return true;
    }
  } while (SkipCommentLine());
  return false;
}

bool Prescanner::FreeFormContinuation() {
  const char *p{at_};
  bool ampersand{*p == '&'};
  if (ampersand) {
    for (++p; *p == ' ' || *p == '\t'; ++p) {
    }
  }
  if (*p != '\n' && (inCharLiteral_ || *p != '!')) {
    return false;
  }
  do {
    if (const char *cont{FreeFormContinuationLine(ampersand)}) {
      BeginSourceLine(cont);
      NextLine();
      return true;
    }
  } while (SkipCommentLine());
  return false;
}

std::optional<Prescanner::LineClassification>
Prescanner::IsFixedFormCompilerDirectiveLine(const char *start) const {
  const char *p{start};
  char col1{*p++};
  if (!IsFixedFormCommentChar(col1)) {
    return {};
  }
  char sentinel[5], *sp{sentinel};
  int column{2};
  for (; column < 6; ++column, ++p) {
    if (*p != ' ') {
      if (*p == '\n' || *p == '\t') {
        break;
      }
      if (sp == sentinel + 1 && sentinel[0] == '$' && IsDecimalDigit(*p)) {
        // OpenMP conditional compilation line: leave the label alone
        break;
      }
      *sp++ = ToLowerCaseLetter(*p);
    }
  }
  if (column == 6) {
    if (*p == ' ' || *p == '\t' || *p == '0') {
      ++p;
    } else {
      // This is a Continuation line, not an initial directive line.
      return {};
    }
  }
  if (sp == sentinel) {
    return {};
  }
  *sp = '\0';
  if (const char *ss{IsCompilerDirectiveSentinel(sentinel)}) {
    std::size_t payloadOffset = p - start;
    return {LineClassification{
        LineClassification::Kind::CompilerDirective, payloadOffset, ss}};
  }
  return {};
}

std::optional<Prescanner::LineClassification>
Prescanner::IsFreeFormCompilerDirectiveLine(const char *start) const {
  char sentinel[8];
  const char *p{start};
  while (*p == ' ' || *p == '\t') {
    ++p;
  }
  if (*p++ != '!') {
    return {};
  }
  for (std::size_t j{0}; j + 1 < sizeof sentinel; ++p, ++j) {
    if (*p == '\n') {
      break;
    }
    if (*p == ' ' || *p == '\t' || *p == '&') {
      if (j == 0) {
        break;
      }
      sentinel[j] = '\0';
      for (++p; *p == ' ' || *p == '\t'; ++p) {
      }
      if (*p == '!') {
        break;
      }
      if (const char *sp{IsCompilerDirectiveSentinel(sentinel)}) {
        std::size_t offset = p - start;
        return {LineClassification{
            LineClassification::Kind::CompilerDirective, offset, sp}};
      }
      break;
    }
    sentinel[j] = ToLowerCaseLetter(*p);
  }
  return {};
}

Prescanner &Prescanner::AddCompilerDirectiveSentinel(const std::string &dir) {
  std::uint64_t packed{0};
  for (char ch : dir) {
    packed = (packed << 8) | (ToLowerCaseLetter(ch) & 0xff);
  }
  compilerDirectiveBloomFilter_.set(packed % prime1);
  compilerDirectiveBloomFilter_.set(packed % prime2);
  compilerDirectiveSentinels_.insert(dir);
  return *this;
}

const char *Prescanner::IsCompilerDirectiveSentinel(
    const char *sentinel) const {
  std::uint64_t packed{0};
  std::size_t n{0};
  for (; sentinel[n] != '\0'; ++n) {
    packed = (packed << 8) | (sentinel[n] & 0xff);
  }
  if (n == 0 || !compilerDirectiveBloomFilter_.test(packed % prime1) ||
      !compilerDirectiveBloomFilter_.test(packed % prime2)) {
    return nullptr;
  }
  const auto iter{compilerDirectiveSentinels_.find(std::string(sentinel, n))};
  return iter == compilerDirectiveSentinels_.end() ? nullptr : iter->data();
}

Prescanner::LineClassification Prescanner::ClassifyLine(
    const char *start) const {
  if (inFixedForm_) {
    if (std::optional<LineClassification> lc{
            IsFixedFormCompilerDirectiveLine(start)}) {
      return std::move(*lc);
    }
    if (IsFixedFormCommentLine(start)) {
      return {LineClassification::Kind::Comment};
    }
  } else {
    if (std::optional<LineClassification> lc{
            IsFreeFormCompilerDirectiveLine(start)}) {
      return std::move(*lc);
    }
    if (IsFreeFormComment(start)) {
      return {LineClassification::Kind::Comment};
    }
  }
  if (std::optional<std::size_t> quoteOffset{IsIncludeLine(start)}) {
    return {LineClassification::Kind::Include, *quoteOffset};
  }
  if (const char *dir{IsPreprocessorDirectiveLine(start)}) {
    if (std::memcmp(dir, "if", 2) == 0 || std::memcmp(dir, "elif", 4) == 0 ||
        std::memcmp(dir, "else", 4) == 0 || std::memcmp(dir, "endif", 5) == 0) {
      return {LineClassification::Kind::ConditionalCompilationDirective};
    } else {
      return {LineClassification::Kind::PreprocessorDirective};
    }
  }
  return {LineClassification::Kind::Source};
}

void Prescanner::SourceFormChange(std::string &&dir) {
  if (dir == "!dir$ free") {
    inFixedForm_ = false;
  } else if (dir == "!dir$ fixed") {
    inFixedForm_ = true;
  }
}

}  // namespace Fortran::parser
