#include "prescan.h"
#include "char-buffer.h"
#include "idioms.h"
#include "source.h"
#include <cctype>
#include <cstring>
#include <utility>
#include <vector>

namespace Fortran {

CharBuffer Prescanner::Prescan(const SourceFile &source) {
  sourceFile_ = &source;
  lineStart_ = source.content();
  limit_ = lineStart_ + source.bytes();
  CharBuffer out;
  TokenSequence tokens, preprocessed;
  while (lineStart_ < limit_) {
    if (CommentLinesAndPreprocessorDirectives() &&
        lineStart_ >= limit_) {
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
      preprocessed.AddChar('\n');
      preprocessed.EndToken();
      if (IsFixedFormCommentLine(preprocessed.data()) ||
          IsFreeFormComment(preprocessed.data())) {
        ++newlineDebt_;
      } else {
        preprocessed.pop_back();  // clip the newline added above
        preprocessed.EmitWithCaseConversion(&out);
      }
      preprocessed.clear();
    } else {
      tokens.EmitWithCaseConversion(&out);
    }
    tokens.clear();
    out.Put('\n');
    PayNewlineDebt(&out);
  }
  PayNewlineDebt(&out);
  sourceFile_ = nullptr;
  return std::move(out);
}

std::optional<TokenSequence> Prescanner::NextTokenizedLine() {
  if (lineStart_ >= limit_) {
    return {};
  }
  bool wasInPreprocessorDirective{inPreprocessorDirective_};
  auto saveAt = at_;
  auto saveAtPosition = atPosition_;
  inPreprocessorDirective_ = true;
  BeginSourceLineAndAdvance();
  TokenSequence tokens;
  while (NextToken(&tokens)) {
  }
  inPreprocessorDirective_ = wasInPreprocessorDirective;
  at_ = saveAt;
  atPosition_ = saveAtPosition;
  return {std::move(tokens)};
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
  lineStartPosition_.AdvanceLine();
}

void Prescanner::LabelField(TokenSequence *token) {
  int outCol{1};
  for (; *at_ != '\n' && atPosition_.column() <= 6; ++at_) {
    if (*at_ == '\t') {
      ++at_;
      atPosition_.set_column(7);
      break;
    }
    if (*at_ != ' ' &&
        (*at_ != '0' ||
         atPosition_.column() != 6)) {  // '0' in column 6 becomes space
      token->AddChar(*at_);
      ++outCol;
    }
    atPosition_.AdvanceColumn();
  }
  if (outCol > 1) {
    token->EndToken();
  }
  if (outCol < 7) {
    for (; outCol < 7; ++outCol) {
      token->AddChar(' ');
    }
    token->EndToken();
  }
}

void Prescanner::NextChar() {
  // CHECK(*at_ != '\n');
  ++at_;
  atPosition_.AdvanceColumn();
  if (inPreprocessorDirective_) {
    while (*at_ == '/' && at_[1] == '*') {
      char star{' '}, slash{' '};
      at_ += 2;
      atPosition_.set_column(atPosition_.column() + 2);
      while ((*at_ != '\n' || slash == '\\') && (star != '*' || slash != '/')) {
        star = slash;
        slash = *at_++;
        atPosition_.AdvanceColumn();
      }
    }
    while (*at_ == '\\' && at_ + 2 < limit_ && at_[1] == '\n') {
      BeginSourceLineAndAdvance();
      ++newlineDebt_;
    }
  } else {
    if ((inFixedForm_ && atPosition_.column() > fixedFormColumnLimit_ &&
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
  if (inFixedForm_) {
    SkipSpaces();
  } else if (*at_ == ' ' || *at_ == '\t') {
    NextChar();
    SkipSpaces();
    if (*at_ != '\n') {
      tokens->AddChar(' ');
      tokens->EndToken();
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
    static constexpr int maxHollerith = 256 * (132-6);
    do {
      if (n < maxHollerith) {
        n = 10 * n + *at_ - '0';
      }
      EmitCharAndAdvance(tokens, *at_);
      if (inFixedForm_) {
        SkipSpaces();
      }
    } while (isdigit(*at_));
    if ((*at_ == 'h' || *at_ == 'H') &&
        n > 0 && n < maxHollerith &&
        !preventHollerith_) {
      EmitCharAndAdvance(tokens, 'h');
      inCharLiteral_ = true;
      while (n-- > 0) {
        if (PadOutCharacterLiteral()) {
          tokens->AddChar(' ');
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
    if (isdigit(EmitCharAndAdvance(tokens, '.'))) {
      while (isdigit(EmitCharAndAdvance(tokens, *at_))) {
      }
      ExponentAndKind(tokens);
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
         (ch == '<' || ch == '>' || ch == '/' || ch == '=' ||
          ch == '!')) ||
        (ch == nch &&
         (ch == '/' || ch == ':' || ch == '*' ||
          ch == '#' || ch == '&' || ch == '|' || ch == '<' || ch == '>')) ||
        (ch == '=' && nch == '>')) {
      // token comprises two characters
      EmitCharAndAdvance(tokens, nch);
    }
  }
  tokens->EndToken();
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
      tokens->AddChar(' ');
    }
    if (*at_ == '\\' && enableBackslashEscapesInCharLiterals_) {
      EmitCharAndAdvance(tokens, '\\');
      while (PadOutCharacterLiteral()) {
        tokens->AddChar(' ');
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
  if (inFixedForm_ &&
      !tabInCurrentLine_ &&
      *at_ == '\n' &&
      atPosition_.column() < fixedFormColumnLimit_) {
    atPosition_.AdvanceColumn();
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
    if (IsFixedFormCommentLine(lineStart_) ||
        IsFreeFormComment(lineStart_)) {
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
    if (IsFixedFormCommentLine(lineStart_) ||
        IsFreeFormComment(lineStart_)) {
      NextLine();
    } else if (IsPreprocessorDirectiveLine(lineStart_)) {
      auto here = lineStartPosition_;
      if (std::optional<TokenSequence> tokens{NextTokenizedLine()}) {
        std::string err{preprocessor_.Directive(*tokens)};
        if (!err.empty()) {
          *error_ << here << ' ' << err << '\n';
        }
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
  if (p[0] == ' ' && p[1] == ' ' && p[2] == ' ' &&
      p[3] == ' ' && p[4] == ' ') {
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
  atPosition_.set_column(7);
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
  (atPosition_ = lineStartPosition_).set_column(column);
  tabInCurrentLine_ = false;
  ++newlineDebt_;
  NextLine();
  return true;
}

void Prescanner::PayNewlineDebt(CharBuffer *out) {
  for (; newlineDebt_ > 0; --newlineDebt_) {
    out->Put('\n');
  }
}
}  // namespace Fortran
