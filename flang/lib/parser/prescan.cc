#include "prescan.h"
#include "characters.h"
#include "idioms.h"
#include "message.h"
#include "preprocessor.h"
#include "source.h"
#include "token-sequence.h"
#include <cstddef>
#include <cstring>
#include <sstream>
#include <utility>
#include <vector>

namespace Fortran {
namespace parser {

Prescanner::Prescanner(
    Messages &messages, CookedSource &cooked, Preprocessor &preprocessor)
  : messages_{messages}, cooked_{cooked}, preprocessor_{preprocessor} {}

Prescanner::Prescanner(const Prescanner &that)
  : messages_{that.messages_}, cooked_{that.cooked_},
    preprocessor_{that.preprocessor_}, inFixedForm_{that.inFixedForm_},
    fixedFormColumnLimit_{that.fixedFormColumnLimit_},
    enableOldDebugLines_{that.enableOldDebugLines_},
    enableBackslashEscapesInCharLiterals_{
        that.enableBackslashEscapesInCharLiterals_},
    warnOnNonstandardUsage_{that.warnOnNonstandardUsage_},
    compilerDirectiveBloomFilter_{that.compilerDirectiveBloomFilter_},
    compilerDirectiveSentinels_{that.compilerDirectiveSentinels_} {}

static void NormalizeCompilerDirectiveCommentMarker(TokenSequence *dir) {
  char *p{dir->GetMutableCharData()};
  char *limit{p + dir->SizeInChars()};
  for (; p < limit; ++p) {
    if (*p != ' ') {
      CHECK(*p == '*' || *p == 'c' || *p == 'C' || *p == '!');
      *p = '!';
      return;
    }
  }
  CHECK(!"compiler directive all blank");
}

bool Prescanner::Prescan(ProvenanceRange range) {
  AllSources &allSources{cooked_.allSources()};
  ProvenanceRange around{allSources.GetContiguousRangeAround(range)};
  startProvenance_ = range.start();
  std::size_t offset{0};
  const SourceFile *source{allSources.GetSourceFile(startProvenance_, &offset)};
  CHECK(source != nullptr);
  start_ = source->content() + offset;
  limit_ = start_ + range.size();
  lineStart_ = start_;
  const bool beganInFixedForm{inFixedForm_};
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
    tokens.Emit(&cooked_);
  }
  return !anyFatalErrors_;
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
  case LineClassification::Kind::PreprocessorDirective:
    if (std::optional<TokenSequence> toks{TokenizePreprocessorDirective()}) {
      preprocessor_.Directive(*toks, this);
    }
    return;
  case LineClassification::Kind::CompilerDirective:
    directiveSentinel_ = line.sentinel;
    CHECK(directiveSentinel_ != nullptr);
    BeginSourceLineAndAdvance();
    if (inFixedForm_) {
      CHECK(*at_ == '!' || *at_ == '*' || *at_ == 'c' || *at_ == 'C');
    } else {
      while (*at_ == ' ' || *at_ == '\t') {
        ++at_;
      }
      CHECK(*at_ == '!');
    }
    tokens.PutNextTokenChar('!', GetCurrentProvenance());
    ++at_, ++column_;
    for (const char *sp{directiveSentinel_}; *sp != '\0';
         ++sp, ++at_, ++column_) {
      tokens.PutNextTokenChar(*sp, GetCurrentProvenance());
    }
    tokens.CloseToken();
    break;
  case LineClassification::Kind::Source:
    BeginSourceLineAndAdvance();
    if (inFixedForm_) {
      LabelField(&tokens);
    } else {
      SkipSpaces();
    }
    break;
  }

  while (NextToken(&tokens)) {
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
    case LineClassification::Kind::PreprocessorDirective:
      Complain("preprocessed line looks like a preprocessor directive"_en_US,
          preprocessed->GetProvenanceRange().start());
      preprocessed->ToLowerCase().Emit(&cooked_);
      break;
    case LineClassification::Kind::CompilerDirective:
      NormalizeCompilerDirectiveCommentMarker(&*preprocessed);
      preprocessed->ToLowerCase();
      SourceFormChange(preprocessed->ToString());
      preprocessed->Emit(&cooked_);
      break;
    case LineClassification::Kind::Source:
      preprocessed->ToLowerCase().Emit(&cooked_);
      break;
    }
  } else {
    tokens.ToLowerCase();
    if (line.kind == LineClassification::Kind::CompilerDirective) {
      SourceFormChange(tokens.ToString());
    }
    tokens.Emit(&cooked_);
    cooked_.Put('\n', newlineProvenance);
  }
  directiveSentinel_ = nullptr;
}

TokenSequence Prescanner::TokenizePreprocessorDirective() {
  CHECK(lineStart_ < limit_ && !inPreprocessorDirective_);
  auto saveAt = at_;
  inPreprocessorDirective_ = true;
  BeginSourceLineAndAdvance();
  TokenSequence tokens;
  while (NextToken(&tokens)) {
  }
  inPreprocessorDirective_ = false;
  at_ = saveAt;
  return {std::move(tokens)};
}

Message &Prescanner::Error(Message &&message) {
  anyFatalErrors_ = true;
  return messages_.Put(std::move(message));
}

Message &Prescanner::Error(MessageFixedText text, Provenance p) {
  anyFatalErrors_ = true;
  return messages_.Put({p, text});
}

Message &Prescanner::Error(MessageFormattedText &&text, Provenance p) {
  anyFatalErrors_ = true;
  return messages_.Put({p, std::move(text)});
}

Message &Prescanner::Complain(Message &&message) {
  return messages_.Put(std::move(message));
}

Message &Prescanner::Complain(MessageFixedText text, Provenance p) {
  return messages_.Put({p, text});
}

Message &Prescanner::Complain(MessageFormattedText &&text, Provenance p) {
  return messages_.Put({p, std::move(text)});
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

void Prescanner::LabelField(TokenSequence *token) {
  int outCol{1};
  for (; *at_ != '\n' && column_ <= 6; ++at_) {
    if (*at_ == '\t') {
      ++at_;
      column_ = 7;
      break;
    }
    if (*at_ != ' ' &&
        (*at_ != '0' || column_ != 6)) {  // '0' in column 6 becomes space
      EmitChar(token, *at_);
      ++outCol;
    }
    ++column_;
  }
  if (outCol > 1) {
    token->CloseToken();
  }
  if (outCol < 7) {
    if (outCol == 1) {
      token->Put("      ", 6, sixSpaceProvenance_.start());
    } else {
      for (; outCol < 7; ++outCol) {
        token->PutNextTokenChar(' ', spaceProvenance_);
      }
      token->CloseToken();
    }
  }
}

void Prescanner::NextChar() {
  CHECK(*at_ != '\n');
  ++at_;
  ++column_;
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
    if ((inFixedForm_ && column_ > fixedFormColumnLimit_ &&
            !tabInCurrentLine_) ||
        (*at_ == '!' && !inCharLiteral_)) {
      // Skip remainder of fixed form line due to '!' comment marker or
      // hitting the right margin.
      while (*at_ != '\n') {
        ++at_;
      }
    }
    while (*at_ == '\n' || *at_ == '&') {
      if (inFixedForm_) {
        if (!FixedFormContinuation()) {
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
  bool wasInCharLiteral{inCharLiteral_};
  inCharLiteral_ = false;
  while (*at_ == ' ' || *at_ == '\t') {
    NextChar();
  }
  inCharLiteral_ = wasInCharLiteral;
}

bool Prescanner::NextToken(TokenSequence *tokens) {
  CHECK(at_ >= start_ && at_ < limit_);
  if (inFixedForm_) {
    SkipSpaces();
  } else if (*at_ == ' ' || *at_ == '\t') {
    // Compress white space into a single space character.
    // Discard white space at the end of a line.
    const auto theSpace = at_;
    NextChar();
    SkipSpaces();
    if (*at_ != '\n') {
      tokens->PutNextTokenChar(' ', GetProvenance(theSpace));
      tokens->CloseToken();
      return true;
    }
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
      if (inFixedForm_ && !inPreprocessorDirective_) {
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
    if (IsDecimalDigit(nch)) {
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
      preventHollerith_ = true;  // ambiguity: CHARACTER*2H
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
    }
  }
  tokens->CloseToken();
  return true;
}

bool Prescanner::ExponentAndKind(TokenSequence *tokens) {
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

void Prescanner::QuotedCharacterLiteral(TokenSequence *tokens) {
  const char *start{at_}, quote{*start};
  inCharLiteral_ = true;
  const auto emit = [&](char ch) { EmitChar(tokens, ch); };
  const auto insert = [&](char ch) { EmitInsertedChar(tokens, ch); };
  bool escape{false};
  while (true) {
    char ch{*at_};
    escape = !escape && ch == '\\' && enableBackslashEscapesInCharLiterals_;
    EmitQuotedChar(
        ch, emit, insert, false, !enableBackslashEscapesInCharLiterals_);
    while (PadOutCharacterLiteral(tokens)) {
    }
    if (*at_ == '\n') {
      if (!inPreprocessorDirective_) {
        Error("incomplete character literal"_en_US, GetProvenance(start));
      }
      break;
    }
    NextChar();
    if (*at_ == quote && !escape) {
      // A doubled quote mark becomes a single instance of the quote character
      // in the literal (later).  There can be spaces between the quotes in
      // fixed form source.
      EmitCharAndAdvance(tokens, quote);
      if (inFixedForm_ && !inPreprocessorDirective_) {
        SkipSpaces();
      }
      if (*at_ != quote) {
        break;
      }
    }
  }
  inCharLiteral_ = false;
}

void Prescanner::Hollerith(TokenSequence *tokens, int count) {
  inCharLiteral_ = true;
  EmitChar(tokens, 'H');
  const char *start{at_};
  while (count-- > 0) {
    if (PadOutCharacterLiteral(tokens)) {
    } else if (*at_ != '\n') {
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
    } else {
      break;
    }
  }
  if (*at_ == '\n') {
    if (!inPreprocessorDirective_) {
      Error("incomplete Hollerith literal"_en_US, GetProvenance(start));
    }
  } else {
    NextChar();
  }
  inCharLiteral_ = false;
}

// In fixed form, source card images must be processed as if they were at
// least 72 columns wide, at least in character literal contexts.
bool Prescanner::PadOutCharacterLiteral(TokenSequence *tokens) {
  while (inFixedForm_ && !tabInCurrentLine_ && at_[1] == '\n') {
    if (column_ < fixedFormColumnLimit_) {
      tokens->PutNextTokenChar(' ', spaceProvenance_);
      ++column_;
      return true;
    }
    if (!FixedFormContinuation() || tabInCurrentLine_) {
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
  if (ch == '*' || ch == 'C' || ch == 'c' ||
      ch == '%' ||  // VAX %list, %eject, &c.
      ((ch == 'D' || ch == 'd') && !enableOldDebugLines_)) {
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

bool Prescanner::FortranInclude(const char *firstQuote) {
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
    Error("malformed path name string"_en_US, GetProvenance(p));
    return true;
  }
  for (++p; *p == ' ' || *p == '\t'; ++p) {
  }
  if (*p != '\n' && *p != '!') {
    Complain("excess characters after path name"_en_US, GetProvenance(p));
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
    Error(MessageFormattedText("INCLUDE: %s"_en_US, error.str().data()),
        provenance);
    return true;
  }
  if (included->bytes() == 0) {
    return true;
  }
  ProvenanceRange includeLineRange{
      provenance, static_cast<std::size_t>(p - lineStart_)};
  ProvenanceRange fileRange{
      allSources.AddIncludedFile(*included, includeLineRange)};
  anyFatalErrors_ |= !Prescanner{*this}.Prescan(fileRange);
  return true;
}

bool Prescanner::IsPreprocessorDirectiveLine(const char *start) const {
  const char *p{start};
  for (; *p == ' '; ++p) {
  }
  if (*p == '#') {
    return !inFixedForm_ || p != start + 5;
  }
  for (; *p == ' ' || *p == '\t'; ++p) {
  }
  return *p == '#';
}

bool Prescanner::IsNextLinePreprocessorDirective() const {
  return IsPreprocessorDirectiveLine(lineStart_);
}

void Prescanner::SkipCommentLines() {
  while (lineStart_ < limit_) {
    LineClassification line{ClassifyLine(lineStart_)};
    if (line.kind != LineClassification::Kind::Comment) {
      break;
    }
    NextLine();
  }
}

const char *Prescanner::FixedFormContinuationLine() {
  const char *p{lineStart_};
  if (p >= limit_) {
    return nullptr;
  }
  tabInCurrentLine_ = false;
  char col1{*p};
  if (directiveSentinel_ != nullptr) {
    // Must be a continued compiler directive.
    if (col1 != '!' && col1 != '*' && col1 != 'c' && col1 != 'C') {
      return nullptr;
    }
    int j{1};
    for (; j < 5; ++j) {
      char ch{directiveSentinel_[j - 1]};
      if (ch == '\0') {
        break;
      }
      if (ch != ToLowerCaseLetter(p[j])) {
        return nullptr;
      }
    }
    for (; j < 5; ++j) {
      if (p[j] != ' ') {
        return nullptr;
      }
    }
    char col6{p[5]};
    if (col6 != '\n' && col6 != '\t' && col6 != ' ' && col6 != '0') {
      return p + 6;
    }
    return nullptr;
  }
  // Normal case: not in a compiler directive.
  if (*p == '&') {
    // Extension: '&' as continuation marker
    if (warnOnNonstandardUsage_) {
      Complain("nonstandard usage"_en_US, GetProvenance(p));
    }
    return p + 1;
  }
  if (*p == '\t' && p[1] >= '1' && p[1] <= '9') {
    tabInCurrentLine_ = true;
    return p + 2;  // VAX extension
  }
  if (p[0] == ' ' && p[1] == ' ' && p[2] == ' ' && p[3] == ' ' && p[4] == ' ') {
    char col6{p[5]};
    if (col6 != '\n' && col6 != '\t' && col6 != ' ' && col6 != '0') {
      return p + 6;
    }
  }
  if (delimiterNesting_ > 0) {
    return p;
  }
  return nullptr;  // not a continuation line
}

bool Prescanner::FixedFormContinuation() {
  // N.B. We accept '&' as a continuation indicator (even) in fixed form.
  if (*at_ == '&' && inCharLiteral_) {
    return false;
  }
  SkipCommentLines();
  const char *cont{FixedFormContinuationLine()};
  if (cont == nullptr) {
    return false;
  }
  BeginSourceLine(cont);
  column_ = 7;
  NextLine();
  return true;
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
  SkipCommentLines();
  p = lineStart_;
  if (p >= limit_) {
    return false;
  }
  for (; *p == ' ' || *p == '\t'; ++p) {
  }
  if (directiveSentinel_ != nullptr) {
    // Look for a continued compiler directive.
    if (*p++ != '!') {
      return false;
    }
    for (const char *s{directiveSentinel_}; *s != '\0'; ++p, ++s) {
      if (*s != ToLowerCaseLetter(*p)) {
        return false;
      }
    }
    for (; *p == ' ' || *p == '\t'; ++p) {
    }
    if (*p == '&') {
      ++p;
    } else if (!ampersand) {
      return false;
    }
  } else {
    // Normal case (not a compiler directive)
    if (*p == '&') {
      ++p;
    } else if (ampersand || delimiterNesting_ > 0) {
      if (p > lineStart_) {
        --p;
      }
    } else {
      return false;  // not a continuation
    }
  }
  at_ = p;
  tabInCurrentLine_ = false;
  NextLine();
  return true;
}

std::optional<Prescanner::LineClassification>
Prescanner::IsFixedFormCompilerDirectiveLine(const char *start) const {
  const char *p{start};
  char col1{*p};
  if (col1 != '*' && col1 != 'C' && col1 != 'c' && col1 != '!') {
    return {};
  }
  char sentinel[5], *sp{sentinel};
  for (int col{2}; col < 6; ++col) {
    char ch{*++p};
    if (ch == '\n' || ch == '\t') {
      return {};
    }
    if (ch != ' ') {
      *sp++ = ToLowerCaseLetter(ch);
    }
  }
  *sp = '\0';
  if (const char *sp{IsCompilerDirectiveSentinel(sentinel)}) {
    return {
        LineClassification{LineClassification::Kind::CompilerDirective, 6, sp}};
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

const char *Prescanner::IsCompilerDirectiveSentinel(const char *s) const {
  std::uint64_t packed{0};
  std::size_t n{0};
  for (; s[n] != '\0'; ++n) {
    packed = (packed << 8) | (s[n] & 0xff);
  }
  if (n == 0 || !compilerDirectiveBloomFilter_.test(packed % prime1) ||
      !compilerDirectiveBloomFilter_.test(packed % prime2)) {
    return nullptr;
  }
  const auto iter = compilerDirectiveSentinels_.find(std::string(s, n));
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
  if (IsPreprocessorDirectiveLine(start)) {
    return {LineClassification::Kind::PreprocessorDirective};
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
}  // namespace parser
}  // namespace Fortran
