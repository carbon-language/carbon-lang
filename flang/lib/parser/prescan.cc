#include "prescan.h"
#include "idioms.h"
#include "source.h"
#include <cctype>
#include <cstring>
#include <utility>
#include <vector>

namespace Fortran {
namespace parser {

Prescanner::Prescanner(Messages *messages, AllSources *allSources)
  : messages_{messages}, allSources_{allSources}, start_{&(*allSources)[0]},
    limit_{start_ + allSources->size()}, preprocessor_{*this} {
  std::string compilerInserts{" ,\"01\n"};
  ProvenanceRange range{allSources->AddCompilerInsertion(compilerInserts)};
  for (size_t j{0}; j < compilerInserts.size(); ++j) {
    compilerInsertionProvenance_[compilerInserts[j]] = range.start + j;
  }
  newlineProvenance_ = CompilerInsertionProvenance('\n');
}

CookedSource Prescanner::Prescan() {
  lineStart_ = start_;
  BeginSourceLine(start_);
  TokenSequence tokens, preprocessed;
  CookedSource cooked{allSources_};
  while (lineStart_ < limit_) {
    if (CommentLinesAndPreprocessorDirectives() && lineStart_ >= limit_) {
      break;
    }
    BeginSourceLineAndAdvance();
    if (inFixedForm_) {
      LabelField(&tokens);
    } else {
      SkipSpaces();
    }
    while (NextToken(&tokens)) {
    }
    if (preprocessor_.MacroReplacement(tokens, &preprocessed)) {
      preprocessed.PutNextTokenChar('\n', newlineProvenance_);
      preprocessed.CloseToken();
      if (IsFixedFormCommentLine(preprocessed.data()) ||
          IsFreeFormComment(preprocessed.data())) {
        ++newlineDebt_;
      } else {
        preprocessed.pop_back();  // clip the newline added above
        preprocessed.EmitWithCaseConversion(&cooked);
      }
      preprocessed.clear();
    } else {
      tokens.EmitWithCaseConversion(&cooked);
    }
    tokens.clear();
    cooked.Put('\n', newlineProvenance_);
    PayNewlineDebt(&cooked);
  }
  PayNewlineDebt(&cooked);
  cooked.Marshal();
  return cooked;
}

std::optional<TokenSequence> Prescanner::NextTokenizedLine() {
  if (lineStart_ >= limit_) {
    return {};
  }
  bool wasInPreprocessorDirective{inPreprocessorDirective_};
  auto saveAt = at_;
  inPreprocessorDirective_ = true;
  BeginSourceLineAndAdvance();
  TokenSequence tokens;
  while (NextToken(&tokens)) {
  }
  inPreprocessorDirective_ = wasInPreprocessorDirective;
  at_ = saveAt;
  return {std::move(tokens)};
}

Provenance Prescanner::CompilerInsertionProvenance(char ch) const {
  return compilerInsertionProvenance_.find(ch)->second;
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
    Provenance provenance{CompilerInsertionProvenance(' ')};
    for (; outCol < 7; ++outCol) {
      token->PutNextTokenChar(' ', provenance);
    }
    token->CloseToken();
  }
}

void Prescanner::NextChar() {
  CHECK(*at_ != '\n');  // TODO pmk
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
      ++newlineDebt_;
    }
  } else {
    if ((inFixedForm_ && column_ > fixedFormColumnLimit_ &&
            !tabInCurrentLine_) ||
        (*at_ == '!' && !inCharLiteral_)) {
      while (*at_ != '\n') {
        ++at_;
      }
    }
    while (*at_ == '\n' || *at_ == '&') {
      if ((inFixedForm_ && !FixedFormContinuation()) ||
          (!inFixedForm_ && !FreeFormContinuation())) {
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
}

static inline bool IsNameChar(char ch) {
  return isalnum(ch) || ch == '_' || ch == '$' || ch == '@';
}

bool Prescanner::NextToken(TokenSequence *tokens) {
  CHECK(at_ >= start_ && at_ < limit_);  // TODO pmk rm?
  if (inFixedForm_) {
    SkipSpaces();
  } else if (*at_ == ' ' || *at_ == '\t') {
    Provenance here{GetCurrentProvenance()};
    NextChar();
    SkipSpaces();
    if (*at_ != '\n') {
      tokens->PutNextTokenChar(' ', here);
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
  } else if (isdigit(*at_)) {
    int n{0};
    static constexpr int maxHollerith = 256 * (132 - 6);
    do {
      if (n < maxHollerith) {
        n = 10 * n + *at_ - '0';
      }
      EmitCharAndAdvance(tokens, *at_);
      if (inFixedForm_) {
        SkipSpaces();
      }
    } while (isdigit(*at_));
    if ((*at_ == 'h' || *at_ == 'H') && n > 0 && n < maxHollerith &&
        !preventHollerith_) {
      EmitCharAndAdvance(tokens, 'h');
      inCharLiteral_ = true;
      while (n-- > 0) {
        if (PadOutCharacterLiteral()) {
          tokens->PutNextTokenChar(' ', compilerInsertionProvenance_[' ']);
        } else {
          if (*at_ == '\n') {
            break;  // TODO error
          }
          EmitCharAndAdvance(tokens, *at_);
        }
      }
      inCharLiteral_ = false;
    } else if (*at_ == '.') {
      while (isdigit(EmitCharAndAdvance(tokens, *at_))) {
      }
      ExponentAndKind(tokens);
    } else if (ExponentAndKind(tokens)) {
    } else if (isalpha(*at_)) {
      // Handles FORMAT(3I9HHOLLERITH) by skipping over the first I so that
      // we don't misrecognize I9HOLLERITH as an identifier in the next case.
      EmitCharAndAdvance(tokens, *at_);
    }
    preventHollerith_ = false;
  } else if (*at_ == '.') {
    char nch{EmitCharAndAdvance(tokens, '.')};
    if (isdigit(nch)) {
      while (isdigit(EmitCharAndAdvance(tokens, *at_))) {
      }
      ExponentAndKind(tokens);
    } else if (nch == '.' && EmitCharAndAdvance(tokens, '.') == '.') {
      EmitCharAndAdvance(tokens, '.');  // variadic macro definition ellipsis
    }
    preventHollerith_ = false;
  } else if (IsNameChar(*at_)) {
    while (IsNameChar(EmitCharAndAdvance(tokens, *at_))) {
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
  char ed = tolower(*at_);
  if (ed != 'e' && ed != 'd') {
    return false;
  }
  EmitCharAndAdvance(tokens, ed);
  if (*at_ == '+' || *at_ == '-') {
    EmitCharAndAdvance(tokens, *at_);
  }
  while (isdigit(*at_)) {
    EmitCharAndAdvance(tokens, *at_);
  }
  if (*at_ == '_') {
    while (IsNameChar(EmitCharAndAdvance(tokens, *at_))) {
    }
  }
  return true;
}

void Prescanner::QuotedCharacterLiteral(TokenSequence *tokens) {
  char quote{*at_};
  inCharLiteral_ = true;
  do {
    EmitCharAndAdvance(tokens, *at_);
    while (PadOutCharacterLiteral()) {
      tokens->PutNextTokenChar(' ', compilerInsertionProvenance_[' ']);
    }
    if (*at_ == '\\' && enableBackslashEscapesInCharLiterals_) {
      EmitCharAndAdvance(tokens, '\\');
      while (PadOutCharacterLiteral()) {
        tokens->PutNextTokenChar(' ', compilerInsertionProvenance_[' ']);
      }
    } else if (*at_ == quote) {
      // A doubled quote mark becomes a single instance of the quote character
      // in the literal later.
      EmitCharAndAdvance(tokens, quote);
      if (inFixedForm_) {
        SkipSpaces();
      }
      if (*at_ != quote) {
        break;
      }
    }
  } while (*at_ != '\n');
  inCharLiteral_ = false;
}

bool Prescanner::PadOutCharacterLiteral() {
  if (inFixedForm_ && !tabInCurrentLine_ && *at_ == '\n' &&
      column_ < fixedFormColumnLimit_) {
    ++column_;
    return true;
  }
  return false;
}

bool Prescanner::IsFixedFormCommentLine(const char *start) {
  if (start >= limit_ || !inFixedForm_) {
    return false;
  }
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

bool Prescanner::IsFreeFormComment(const char *p) {
  if (p >= limit_ || inFixedForm_) {
    return false;
  }
  while (*p == ' ' || *p == '\t') {
    ++p;
  }
  return *p == '!' || *p == '\n';
}

bool Prescanner::IsPreprocessorDirectiveLine(const char *start) {
  const char *p{start};
  if (p >= limit_ || inPreprocessorDirective_) {
    return false;
  }
  for (; *p == ' '; ++p) {
  }
  if (*p == '#') {
    return !inFixedForm_ || p != start + 5;
  }
  for (; *p == ' ' || *p == '\t'; ++p) {
  }
  return *p == '#';
}

bool Prescanner::CommentLines() {
  bool any{false};
  while (lineStart_ < limit_) {
    if (IsFixedFormCommentLine(lineStart_) || IsFreeFormComment(lineStart_)) {
      NextLine();
      ++newlineDebt_;
      any = true;
    } else {
      break;
    }
  }
  return any;
}

bool Prescanner::CommentLinesAndPreprocessorDirectives() {
  bool any{false};
  while (lineStart_ < limit_) {
    if (IsFixedFormCommentLine(lineStart_) || IsFreeFormComment(lineStart_)) {
      NextLine();
    } else if (IsPreprocessorDirectiveLine(lineStart_)) {
      if (std::optional<TokenSequence> tokens{NextTokenizedLine()}) {
        anyFatalErrors_ |= !preprocessor_.Directive(*tokens);
      }
    } else {
      break;
    }
    ++newlineDebt_;
    any = true;
  }
  return any;
}

const char *Prescanner::FixedFormContinuationLine() {
  const char *p{lineStart_};
  if (p >= limit_) {
    return nullptr;
  }
  tabInCurrentLine_ = false;
  if (*p == '&') {
    return p + 1;  // extension
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
  CommentLines();
  const char *cont{FixedFormContinuationLine()};
  if (cont == nullptr) {
    return false;
  }
  BeginSourceLine(cont);
  column_ = 7;
  ++newlineDebt_;
  NextLine();
  return true;
}

bool Prescanner::FreeFormContinuation() {
  while (*at_ == ' ' || *at_ == '\t') {
    ++at_;
  }
  const char *p{at_};
  bool ampersand{*p == '&'};
  if (ampersand) {
    for (++p; *p == ' ' || *p == '\t'; ++p) {
    }
  }
  if (*p != '\n' && (inCharLiteral_ || *p != '!')) {
    return false;
  }
  CommentLines();
  p = lineStart_;
  if (p >= limit_) {
    return false;
  }
  int column{1};
  for (; *p == ' ' || *p == '\t'; ++p) {
    ++column;
  }
  if (*p == '&') {
    ++p;
    ++column;
  } else if (ampersand || delimiterNesting_ > 0) {
    if (p > lineStart_) {
      --p;
      --column;
    }
  } else {
    return false;  // not a continuation
  }
  at_ = p;
  column_ = column;
  tabInCurrentLine_ = false;
  ++newlineDebt_;
  NextLine();
  return true;
}

void Prescanner::PayNewlineDebt(CookedSource *cooked) {
  for (; newlineDebt_ > 0; --newlineDebt_) {
    cooked->Put('\n', newlineProvenance_);
  }
}
}  // namespace parser
}  // namespace Fortran
