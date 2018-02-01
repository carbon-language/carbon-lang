#include "preprocessor.h"
#include "char-buffer.h"
#include "idioms.h"
#include "prescan.h"
#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <ctime>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <utility>

namespace Fortran {

void TokenSequence::Append(const TokenSequence &that) {
  if (nextStart_ < char_.size()) {
    start_.push_back(nextStart_);
  }
  int offset = char_.size();
  for (int st : that.start_) {
    start_.push_back(st + offset);
  }
  char_.insert(char_.end(), that.char_.begin(), that.char_.end());
  nextStart_ = char_.size();
}

void TokenSequence::EmitWithCaseConversion(CharBuffer *out) {
  size_t tokens{start_.size()};
  size_t chars{char_.size()};
  size_t atToken{0};
  for (size_t j{0}; j < chars; ) {
    size_t nextStart{atToken + 1 < tokens ? start_[++atToken] : chars};
    if (isalpha(char_[j])) {
      for (; j < nextStart; ++j) {
        out->Put(tolower(char_[j]));
      }
    } else {
      out->Put(&char_[j], nextStart - j);
      j = nextStart;
    }
  }
}

Definition::Definition(const TokenSequence &repl, size_t firstToken,
                       size_t tokens)
  : replacement_{Tokenize({}, repl, firstToken, tokens)} {}

Definition::Definition(const std::vector<std::string> &argNames,
                       const TokenSequence &repl, size_t firstToken,
                       size_t tokens)
  : isFunctionLike_{true}, argumentCount_(argNames.size()),
    replacement_{Tokenize(argNames, repl, firstToken, tokens)} {}

Definition::Definition(const std::string &predefined)
  : isPredefined_{true}, replacement_{predefined} {}

bool Definition::set_isDisabled(bool disable) {
  bool was{isDisabled_};
  isDisabled_ = disable;
  return was;
}

static bool IsIdentifierFirstCharacter(char ch) {
  return ch == '_' || isalpha(ch);
}

static bool IsIdentifierFirstCharacter(const CharPointerWithLength &cpl) {
  return cpl.size() > 0 && IsIdentifierFirstCharacter(cpl[0]);
}

TokenSequence Definition::Tokenize(const std::vector<std::string> &argNames,
                                   const TokenSequence &token,
                                   size_t firstToken, size_t tokens) {
  std::map<std::string, std::string> args;
  char argIndex{'A'};
  for (const std::string &arg : argNames) {
    CHECK(args.find(arg) == args.end());
    args[arg] = "~"s + argIndex++;
  }
  TokenSequence result;
  for (size_t j{0}; j < tokens; ++j) {
    size_t bytes{token.GetBytes(firstToken + j)};
    if (bytes == 0) {
      continue;
    }
    const char *text{token.GetText(firstToken + j)};
    if (bytes > 0 && IsIdentifierFirstCharacter(*text)) {
      auto it = args.find(token.GetString(firstToken + j));
      if (it != args.end()) {
        result.push_back(it->second);
        continue;
      }
    }
    result.push_back(text, bytes);
  }
  return result;
}

static bool IsBlank(const CharPointerWithLength &cpl) {
  size_t bytes{cpl.size()};
  for (size_t j{0}; j < bytes; ++j) {
    char ch{cpl[j]};
    if (ch != ' ' && ch != '\t') {
      return false;
    }
  }
  return true;
}

TokenSequence Definition::Apply(const std::vector<TokenSequence> &args) {
  TokenSequence result;
  bool stringify{false}, pasting{false};
  size_t tokens{replacement_.size()};
  for (size_t j{0}; j < tokens; ++j) {
    const CharPointerWithLength &token{replacement_[j]};
    size_t bytes{token.size()};
    const char *text{token.data()};
    if (bytes == 2 && *text == '~') {
      size_t index = text[1] - 'A';
      if (index >= args.size()) {
        continue;
      }
      size_t argTokens{args[index].size()};
      if (stringify) {
        std::string strung{'"'};
        for (size_t k{0}; k < argTokens; ++k) {
          size_t argBytes{args[index].GetBytes(k)};
          const char *arg{args[index].GetText(k)};
          for (size_t n{0}; n < argBytes; ++n) {
            char ch{arg[n]};
            if (ch == '"' || ch == '\\') {
              strung += '\\';
            }
            strung += ch;
          }
        }
        strung += '"';
        result.pop_back();  // remove the '#'
        result.push_back(strung);
      } else {
        for (size_t k{0}; k < argTokens; ++k) {
          const CharPointerWithLength &argToken{args[index][k]};
          if (pasting && IsBlank(argToken)) {
          } else {
            result.push_back(argToken);
            pasting = false;
          }
        }
      }
    } else if (bytes == 2 && text[0] == '#' && text[1] == '#') {
      // Token pasting operator in body (not expanded argument); discard any
      // immediately preceding white space, then reopen the last token.
      while (!result.empty() && IsBlank(result[result.size() - 1])) {
        result.pop_back();
      }
      if (!result.empty()) {
        result.ReopenLastToken();
        pasting = true;
      }
    } else if (pasting && IsBlank(token)) {
      // Delete whitespace immediately following ## in the body.
    } else {
      stringify = bytes == 1 && *text == '#';
      result.push_back(text, bytes);
      pasting = false;
    }
  }
  return result;
}

static std::string FormatTime(const std::time_t &now, const char *format) {
  char buffer[16];
  return {buffer,
          std::strftime(buffer, sizeof buffer, format, std::localtime(&now))};
}

Preprocessor::Preprocessor(Prescanner &ps) : prescanner_{ps} {
  // Capture current local date & time once now to avoid having the values
  // of __DATE__ or __TIME__ change during compilation.
  std::time_t now;
  std::time(&now);
  definitions_.emplace(SaveToken("__DATE__"s),  // e.g., "Jun 16 1904"
                       Definition{FormatTime(now, "\"%h %e %Y\""), 0, 1});
  definitions_.emplace(SaveToken("__TIME__"s),  // e.g., "23:59:60"
                       Definition{FormatTime(now, "\"%T\""), 0, 1});
  // The values of these predefined macros depend on their invocation sites.
  definitions_.emplace(SaveToken("__FILE__"s), Definition{"__FILE__"s});
  definitions_.emplace(SaveToken("__LINE__"s), Definition{"__LINE__"s});
}

bool Preprocessor::MacroReplacement(const TokenSequence &input,
                                    TokenSequence *result) {
  // Do quick scan for any use of a defined name.
  size_t tokens{input.size()};
  size_t j;
  for (j = 0; j < tokens; ++j) {
    size_t bytes{input.GetBytes(j)};
    if (bytes > 0 &&
        IsIdentifierFirstCharacter(*input.GetText(j)) &&
        IsNameDefined(input[j])) {
      break;
    }
  }
  if (j == tokens) {
    return false;  // nothing appeared that could be replaced
  }

  for (size_t k{0}; k < j; ++k) {
    result->push_back(input[k]);
  }
  for (; j < tokens; ++j) {
    const CharPointerWithLength &token{input[j]};
    if (IsBlank(token) || !IsIdentifierFirstCharacter(token[0])) {
      result->push_back(token);
      continue;
    }
    auto it = definitions_.find(token);
    if (it == definitions_.end()) {
      result->push_back(token);
      continue;
    }
    Definition &def{it->second};
    if (def.isDisabled()) {
      result->push_back(token);
      continue;
    }
    if (!def.isFunctionLike()) {
      if (def.isPredefined()) {
        std::string name{def.replacement()[0].ToString()};
        if (name == "__FILE__") {
          result->Append("\""s + prescanner_.sourceFile().path() + '"');
          continue;
        }
        if (name == "__LINE__") {
          std::stringstream ss;
          ss << prescanner_.position().lineNumber();
          result->Append(ss.str());
          continue;
        }
      }
      def.set_isDisabled(true);
      result->Append(ReplaceMacros(def.replacement()));
      def.set_isDisabled(false);
      continue;
    }
    // Possible function-like macro call.  Skip spaces and newlines to see
    // whether '(' is next.
    size_t k{j};
    bool leftParen{false};
    while (++k < tokens) {
      const CharPointerWithLength &lookAhead{input[k]};
      if (!IsBlank(lookAhead) && lookAhead[0] != '\n') {
        leftParen = lookAhead[0] == '(' && lookAhead.size() == 1;
        break;
      }
    }
    if (!leftParen) {
      result->push_back(token);
      continue;
    }
    std::vector<size_t> argStart{++k};
    for (int nesting{0}; k < tokens; ++k) {
      if (input.GetBytes(k) == 1) {
        char ch{*input.GetText(k)};
        if (ch == '(') {
          ++nesting;
        } else if (ch == ')') {
          if (nesting == 0) {
            break;
          }
          --nesting;
        } else if (ch == ',' && nesting == 0) {
          argStart.push_back(k + 1);
        }
      }
    }
    if (k >= tokens ||
        argStart.size() != def.argumentCount()) {
      result->push_back(token);
      continue;
    }
    j = k;  // advance to the terminal ')'
    std::vector<TokenSequence> args;
    for (k = 0; k < argStart.size(); ++k) {
      size_t at{argStart[k]};
      size_t count{(k + 1 == argStart.size() ? j : argStart[k+1] - 1) - at};
      TokenSequence actual;
      for (; count-- > 0; ++at) {
        actual.push_back(input.GetText(at), input.GetBytes(at));
      }
      args.emplace_back(std::move(actual));
    }
    def.set_isDisabled(true);
    result->Append(ReplaceMacros(def.Apply(args)));
    def.set_isDisabled(false);
  }
  return true;
}

TokenSequence Preprocessor::ReplaceMacros(const TokenSequence &tokens) {
  TokenSequence repl;
  return MacroReplacement(tokens, &repl) ? repl : tokens;
}

static size_t SkipBlanks(const TokenSequence &tokens, size_t at,
                         size_t lastToken) {
  for (; at < lastToken; ++at) {
    if (!IsBlank(tokens[at])) {
      break;
    }
  }
  return std::min(at, lastToken);
}

static TokenSequence StripBlanks(const TokenSequence &token, size_t first,
                                 size_t tokens) {
  TokenSequence noBlanks;
  for (size_t j{SkipBlanks(token, first, tokens)}; j < tokens;
       j = SkipBlanks(token, j + 1, tokens)) {
    noBlanks.push_back(token[j]);
  }
  return noBlanks;
}

static std::string ConvertToLowerCase(const std::string &str) {
  std::string lowered{str};
  for (char &ch : lowered) {
    ch = tolower(ch);
  }
  return lowered;
}

static std::string GetDirectiveName(const TokenSequence &line, size_t *rest) {
  size_t tokens{line.size()};
  size_t j{SkipBlanks(line, 0, tokens)};
  if (j == tokens || line.GetString(j) != "#") {
    *rest = tokens;
    return {};
  }
  j = SkipBlanks(line, j + 1, tokens);
  if (j == tokens) {
    *rest = tokens;
    return {};
  }
  *rest = SkipBlanks(line, j + 1, tokens);
  return ConvertToLowerCase(line.GetString(j));
}

std::string Preprocessor::Directive(const TokenSequence &dir) {
  size_t tokens{dir.size()};
  size_t j{SkipBlanks(dir, 0, tokens)};
  if (j == tokens) {
    return {};
  }
  if (dir.GetString(j) != "#") {
    return "missing '#'";
  }
  j = SkipBlanks(dir, j + 1, tokens);
  if (j == tokens) {
    return {};
  }
  if (isdigit(*dir.GetText(j)) || *dir.GetText(j) == '"') {
    return {};  // TODO: treat as #line
  }
  std::string dirName{ConvertToLowerCase(dir.GetString(j))};
  j = SkipBlanks(dir, j + 1, tokens);
  CharPointerWithLength nameToken;
  if (j < tokens && IsIdentifierFirstCharacter(*dir.GetText(j))) {
    nameToken = dir[j];
  }
  if (dirName == "line") {
    // TODO
    return {};
  }
  if (dirName == "define") {
    if (nameToken.empty()) {
      return "#define: missing or invalid name";
    }
    nameToken = SaveToken(nameToken);
    definitions_.erase(nameToken);
    if (++j < tokens && dir.GetBytes(j) == 1 && *dir.GetText(j) == '(') {
      j = SkipBlanks(dir, j + 1, tokens);
      std::vector<std::string> argName;
      if (dir.GetString(j) != ")") {
        while (true) {
          std::string an{dir.GetString(j)};
          if (an.empty() || !IsIdentifierFirstCharacter(an[0])) {
            return "#define: missing or invalid argument name";
          }
          argName.push_back(an);
          j = SkipBlanks(dir, j + 1, tokens);
          if (j == tokens) {
            return "#define: malformed argument list";
          }
          std::string punc{dir.GetString(j)};
          if (punc == ")") {
            break;
          }
          if (punc != ",") {
            return "#define: malformed argument list";
          }
          j = SkipBlanks(dir, j + 1, tokens);
          if (j == tokens) {
            return "#define: malformed argument list";
          }
        }
        if (std::set<std::string>(argName.begin(), argName.end()).size() !=
            argName.size()) {
          return "#define: argument names are not distinct";
        }
      }
      j = SkipBlanks(dir, j + 1, tokens);
      definitions_.emplace(
        std::make_pair(nameToken, Definition{argName, dir, j, tokens - j}));
    } else {
      definitions_.emplace(
        std::make_pair(nameToken, Definition{dir, j, tokens - j}));
    }
    return {};
  }
  if (dirName == "undef") {
    if (nameToken.empty()) {
      return "#undef: missing or invalid name";
    }
    j = SkipBlanks(dir, j + 1, tokens);
    if (j != tokens) {
      return "#undef: excess tokens at end of directive";
    }
    definitions_.erase(nameToken);
    return {};
  }
  if (dirName == "ifdef" || dirName == "ifndef") {
    if (nameToken.empty()) {
      return "#"s + dirName + ": missing name";
    }
    j = SkipBlanks(dir, j + 1, tokens);
    if (j != tokens) {
      return "#"s + dirName + ": excess tokens at end of directive";
    }
    if (IsNameDefined(nameToken) == (dirName == "ifdef")) {
      ifStack_.push(CanDeadElseAppear::Yes);
      return {};
    }
    return SkipDisabledConditionalCode(dirName, IsElseActive::Yes);
  }
  if (dirName == "if") {
    std::string errors;
    if (IsIfPredicateTrue(dir, j, tokens - j, &errors) || !errors.empty()) {
      ifStack_.push(CanDeadElseAppear::Yes);
    } else {
      errors = SkipDisabledConditionalCode(dirName, IsElseActive::Yes);
    }
    return errors.empty() ? ""s : "#if: "s + errors;
  }
  if (dirName == "else") {
    if (j != tokens) {
      return "#else: excess tokens at end of directive";
    }
    if (ifStack_.empty()) {
      return "#else: not nested within #if, #ifdef, or #ifndef";
    }
    if (ifStack_.top() != CanDeadElseAppear::Yes) {
      return "#else: already appeared within this #if, #ifdef, or #ifndef";
    }
    ifStack_.pop();
    return SkipDisabledConditionalCode("else", IsElseActive::No);
  }
  if (dirName == "elif") {
    if (ifStack_.empty()) {
      return "#elif: not nested within #if, #ifdef, or #ifndef";
    }
    if (ifStack_.top() != CanDeadElseAppear::Yes) {
      return "#elif: #else previously appeared within this "
             "#if, #ifdef, or #ifndef";
    }
    ifStack_.pop();
    return SkipDisabledConditionalCode("elif", IsElseActive::No);
  }
  if (dirName == "endif") {
    if (j != tokens) {
      return "#endif: excess tokens at end of directive";
    }
    if (ifStack_.empty()) {
      return "#endif: no #if, #ifdef, or #ifndef";
    }
    ifStack_.pop();
    return {};
  }
  if (dirName == "error" || dirName == "warning") {
    return {dir.data(), dir.size()};
  }
  return "#"s + dirName + ": unknown or unimplemented directive";
}

CharPointerWithLength Preprocessor::SaveToken(const CharPointerWithLength &t) {
  names_.push_back(t.ToString());
  return {names_.back().data(), names_.back().size()};
}

bool Preprocessor::IsNameDefined(const CharPointerWithLength &token) {
  return definitions_.find(token) != definitions_.end();
}

std::string
Preprocessor::SkipDisabledConditionalCode(const std::string &dirName,
                                          IsElseActive isElseActive) {
  int nesting{0};
  while (std::optional<TokenSequence> line{prescanner_.NextTokenizedLine()}) {
    size_t rest{0};
    std::string dn{GetDirectiveName(*line, &rest)};
    if (dn == "ifdef" || dn == "ifndef" || dn == "if") {
      ++nesting;
    } else if (dn == "endif") {
      if (nesting-- == 0) {
        return {};
      }
    } else if (isElseActive == IsElseActive::Yes && nesting == 0) {
      if (dn == "else") {
        ifStack_.push(CanDeadElseAppear::No);
        return {};
      }
      if (dn == "elif") {
        std::string errors;
        if (IsIfPredicateTrue(*line, rest, line->size() - rest, &errors) ||
            !errors.empty()) {
          ifStack_.push(CanDeadElseAppear::No);
          return errors.empty() ? ""s : "#elif: "s + errors;
        }
      }
    }
  }
  return "#"s + dirName + ": missing #endif";
}

// Precedence level codes used here to accommodate mixed Fortran and C:
// 13: parentheses and constants, logical !, bitwise ~
// 12: unary + and -
// 11: **
// 10: *, /, % (modulus)
//  9: + and -
//  8: << and >>
//  7: bitwise &
//  6: bitwise ^
//  5: bitwise |
//  4: relations (.EQ., ==, &c.)
//  3: .NOT.
//  2: .AND., &&
//  1: .OR., ||
//  0: .EQV. and .NEQV. / .XOR.
// TODO: Ternary and comma operators?
static std::int64_t ExpressionValue(const TokenSequence &token,
                                    int minimumPrecedence,
                                    size_t *atToken, std::string *errors) {
  enum Operator {
    PARENS, CONST, NOTZERO /*!*/, COMPLEMENT /*~*/, UPLUS, UMINUS, POWER,
    TIMES, DIVIDE, MODULUS, ADD, SUBTRACT, LEFTSHIFT, RIGHTSHIFT,
    BITAND, BITXOR, BITOR,
    LT, LE, EQ, NE, GE, GT,
    NOT, AND, OR, EQV, NEQV
  };
  static const int precedence[]{
    13, 13, 13, 13,  // (), 0, !, ~
    12, 12,  // unary +, -
    11, 10, 10, 10, 9, 9, 8, 8,  // **, *, /, %, +, -, <<, >>
    7, 6, 5,  // &, ^, |
    4, 4, 4, 4, 4, 4,  // relations
    3, 2, 1, 0, 0  // .NOT., .AND., .OR., .EQV., .NEQV.
  };
  static const int operandPrecedence[]{
    0, -1, 13, 13,
    13, 13,
    11, 10, 10, 10, 9, 9, 9, 9,
    7, 6, 5,
    5, 5, 5, 5, 5, 5,
    4, 2, 1, 1, 1
  };

  static std::map<std::string, enum Operator> opNameMap;
  if (opNameMap.empty()) {
    opNameMap["("] = PARENS;
    opNameMap["!"] = NOTZERO;
    opNameMap["~"] = COMPLEMENT;
    opNameMap["**"] = POWER;
    opNameMap["*"] = TIMES;
    opNameMap["/"] = DIVIDE;
    opNameMap["%"] = MODULUS;
    opNameMap["+"] = ADD;
    opNameMap["-"] = SUBTRACT;
    opNameMap["<<"] = LEFTSHIFT;
    opNameMap[">>"] = RIGHTSHIFT;
    opNameMap["&"] = BITAND;
    opNameMap["^"] = BITXOR;
    opNameMap["|"] = BITOR;
    opNameMap[".lt."] = opNameMap["<"] = LT;
    opNameMap[".le."] = opNameMap["<="] = LE;
    opNameMap[".eq."] = opNameMap["=="] = EQ;
    opNameMap[".ne."] = opNameMap["/="] = opNameMap["!="] = NE;
    opNameMap[".ge."] = opNameMap[">="] = GE;
    opNameMap[".gt."] = opNameMap[">"] = GT;
    opNameMap[".not."] = NOT;
    opNameMap[".and."] = opNameMap[".a."] = opNameMap["&&"] = AND;
    opNameMap[".or."] = opNameMap[".o."] = opNameMap["||"] = OR;
    opNameMap[".eqv."] = EQV;
    opNameMap[".neqv."] = opNameMap[".xor."] = opNameMap[".x."] = NEQV;
  }

  size_t tokens{token.size()};
  if (*atToken >= tokens) {
    *errors = "incomplete expression";
    return 0;
  }
  std::string t{token.GetString(*atToken)};
  enum Operator op;

  // Parse and evaluate a primary or a unary operator and its operand.
  std::int64_t left{0};
  if (t == "(") {
    op = PARENS;
  } else if (isdigit(t[0])) {
    op = CONST;
    size_t consumed{0};
    left = std::stoll(t, &consumed);
    if (consumed < t.size()) {
      *errors = "uninterpretable numeric constant '"s + t + '\'';
    }
  } else if (IsIdentifierFirstCharacter(t[0])) {
    // undefined macro name -> zero
    // TODO: BOZ constants?
    op = CONST;
  } else if (t == "+") {
    op = UPLUS;
  } else if (t == "-") {
    op = UMINUS;
  } else if (t == "." && *atToken + 2 < tokens &&
             ConvertToLowerCase(token.GetString(*atToken + 1)) == "not" &&
             token.GetString(*atToken + 2) == ".") {
    op = NOT;
    *atToken += 2;
  } else {
    auto it = opNameMap.find(t);
    if (it != opNameMap.end()) {
      op = it->second;
    } else {
      *errors = "operand expected in expression";
      return 0;
    }
  }
  if (precedence[op] < minimumPrecedence && errors->empty()) {
    *errors = "operator precedence error";
  }
  ++*atToken;
  if (op != CONST && errors->empty()) {
    left = ExpressionValue(token, operandPrecedence[op], atToken, errors);
    switch (op) {
    case PARENS:
      if (*atToken < tokens && token.GetString(*atToken) == ")") {
        ++*atToken;
      } else if (errors->empty()) {
        *errors = "')' missing from expression";
      }
      break;
    case NOTZERO:
      left = !left;
      break;
    case COMPLEMENT:
      left = ~left;
      break;
    case UPLUS:
      break;
    case UMINUS:
      left = -left;
      break;
    case NOT:
      left = -!left;
      break;
    DEFAULT_CRASH;
    }
  }
  if (!errors->empty() || *atToken >= tokens) {
    return left;
  }

  // Parse and evaluate a binary operator and its second operand, if present.
  int advance{1};
  t = token.GetString(*atToken);
  if (t == "." && *atToken + 2 < tokens &&
      token.GetString(*atToken + 2) == ".") {
    t += ConvertToLowerCase(token.GetString(*atToken + 1)) + '.';
    advance = 3;
  }
  auto it = opNameMap.find(t);
  if (it == opNameMap.end()) {
    return left;
  }
  op = it->second;
  if (precedence[op] < minimumPrecedence) {
    return left;
  }
  *atToken += advance;
  std::int64_t right{ExpressionValue(token, operandPrecedence[op],
                                     atToken, errors)};
  switch (op) {
  case POWER:
    if (left == 0 && right < 0) {
      *errors = "0 ** negative power";
    }
    if (left == 0 || left == 1 || right == 1) {
      return left;
    }
    if (right <= 0) {
      return !right;
    }
    { std::int64_t power{1};
      for (; right > 0; --right) {
        if ((power * left) / left != power) {
          *errors = "overflow in exponentation";
          return 0;
        }
        power *= left;
      }
      return power;
    }
  case TIMES:
    if (left == 0 || right == 0) {
      return 0;
    }
    if ((left * right) / left != right) {
      *errors = "overflow in multiplication";
    }
    return left * right;
  case DIVIDE:
    if (right == 0) {
      *errors = "division by zero";
      return 0;
    }
    return left / right;
  case MODULUS:
    if (right == 0) {
      *errors = "modulus by zero";
      return 0;
    }
    return left % right;
  case ADD:
    if ((left < 0) == (right < 0) &&
        (left < 0) != (left + right < 0)) {
      *errors = "overflow in addition";
    }
    return left + right;
  case SUBTRACT:
    if ((left < 0) != (right < 0) &&
        (left < 0) == (left - right < 0)) {
      *errors = "overflow in subtraction";
    }
    return left - right;
  case LEFTSHIFT:
    if (right < 0 || right > 64) {
      *errors = "bad left shift count";
    }
    return right >= 64 ? 0 : left << right;
  case RIGHTSHIFT:
    if (right < 0 || right > 64) {
      *errors = "bad right shift count";
    }
    return right >= 64 ? 0 : left >> right;
  case BITAND:
  case AND:
    return left & right;
  case BITXOR:
    return left ^ right;
  case BITOR:
  case OR:
    return left | right;
  case LT:
    return -(left < right);
  case LE:
    return -(left <= right);
  case EQ:
    return -(left == right);
  case NE:
    return -(left != right);
  case GE:
    return -(left >= right);
  case GT:
    return -(left > right);
  case EQV:
    return -(!left == !right);
  case NEQV:
    return -(!left != !right);
  DEFAULT_CRASH;
  }
  return 0;  // silence compiler warning
}

bool
Preprocessor::IsIfPredicateTrue(const TokenSequence &expr, size_t first,
                                size_t exprTokens, std::string *errors) {
  TokenSequence expr1{StripBlanks(expr, first, first + exprTokens)};
  TokenSequence expr2;
  for (size_t j{0}; j < expr1.size(); ++j) {
    if (ConvertToLowerCase(expr1.GetString(j)) == "defined") {
      CharPointerWithLength name;
      if (j + 3 < expr1.size() &&
          expr1.GetString(j + 1) == "(" &&
          expr1.GetString(j + 3) == ")") {
        name = expr1[j + 2];
        j += 3;
      } else if (j + 1 < expr1.size() &&
                 IsIdentifierFirstCharacter(expr1[j + 1])) {
        name = expr1[j++];
      }
      if (!name.empty()) {
        expr2.push_back(IsNameDefined(name) ? "1" : "0", 1);
        continue;
      }
    }
    expr2.push_back(expr1[j]);
  }
  TokenSequence expr3{ReplaceMacros(expr2)};
  TokenSequence expr4{StripBlanks(expr3, 0, expr3.size())};
  size_t atToken{0};
  bool result{ExpressionValue(expr4, 0, &atToken, errors) != 0};
  if (atToken < expr4.size() && errors->empty()) {
    *errors = atToken == 0 ? "could not parse any expression"
                           : "excess characters after expression";
  }
  return result;
}
}  // namespace Fortran
