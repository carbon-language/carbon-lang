#include "preprocessor.h"
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
#include <iostream>  // TODO pmk rm

namespace Fortran {
namespace parser {

Definition::Definition(
    const TokenSequence &repl, size_t firstToken, size_t tokens)
  : replacement_{Tokenize({}, repl, firstToken, tokens)} {}

Definition::Definition(const std::vector<std::string> &argNames,
    const TokenSequence &repl, size_t firstToken, size_t tokens,
    bool isVariadic)
  : isFunctionLike_{true},
    argumentCount_(argNames.size()), isVariadic_{isVariadic},
    replacement_{Tokenize(argNames, repl, firstToken, tokens)} {}

Definition::Definition(const std::string &predefined, AllSources *sources)
  : isPredefined_{true},
    replacement_{predefined,
        sources->AddCompilerInsertion(predefined).LocalOffsetToProvenance(0)} {}

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
    const TokenSequence &token, size_t firstToken, size_t tokens) {
  std::map<std::string, std::string> args;
  char argIndex{'A'};
  for (const std::string &arg : argNames) {
    CHECK(args.find(arg) == args.end());
    args[arg] = "~"s + argIndex++;
  }
  TokenSequence result;
  for (size_t j{0}; j < tokens; ++j) {
    CharPointerWithLength tok{token[firstToken + j]};
    if (IsIdentifierFirstCharacter(tok)) {
      auto it = args.find(tok.ToString());
      if (it != args.end()) {
        result.Put(it->second, token.GetTokenProvenance(j));
        continue;
      }
    }
    result.Put(token, firstToken + j, 1);
  }
  return result;
}

TokenSequence Definition::Apply(
    const std::vector<TokenSequence> &args, const AllSources &allSources) {
  TokenSequence result;
  bool pasting{false};
  bool skipping{false};
  int parenthesesNesting{0};
  size_t tokens{replacement_.size()};
  for (size_t j{0}; j < tokens; ++j) {
    const CharPointerWithLength &token{replacement_[j]};
    size_t bytes{token.size()};
    if (skipping) {
      if (bytes == 1) {
        if (token[0] == '(') {
          ++parenthesesNesting;
        } else if (token[0] == ')') {
          skipping = --parenthesesNesting > 0;
        }
      }
      continue;
    }
    if (bytes == 2 && token[0] == '~') {
      size_t index = token[1] - 'A';
      if (index >= args.size()) {
        continue;
      }
      size_t afterLastNonBlank{result.size()};
      for (; afterLastNonBlank > 0; --afterLastNonBlank) {
        if (!result[afterLastNonBlank - 1].IsBlank()) {
          break;
        }
      }
      size_t argTokens{args[index].size()};
      if (afterLastNonBlank > 0 &&
          result[afterLastNonBlank - 1].ToString() == "#") {
        while (result.size() >= afterLastNonBlank) {
          result.pop_back();
        }
        Provenance quoteProvenance{allSources.CompilerInsertionProvenance('"')};
        result.PutNextTokenChar('"', quoteProvenance);
        for (size_t k{0}; k < argTokens; ++k) {
          const CharPointerWithLength &arg{args[index][k]};
          size_t argBytes{args[index][k].size()};
          for (size_t n{0}; n < argBytes; ++n) {
            char ch{arg[n]};
            Provenance from{args[index].GetTokenProvenance(k, n)};
            if (ch == '"' || ch == '\\') {
              result.PutNextTokenChar(ch, from);
            }
            result.PutNextTokenChar(ch, from);
          }
        }
        result.PutNextTokenChar('"', quoteProvenance);
        result.CloseToken();
      } else {
        for (size_t k{0}; k < argTokens; ++k) {
          if (!pasting || !args[index][k].IsBlank()) {
            result.Put(args[index], k);
            pasting = false;
          }
        }
      }
    } else if (bytes == 2 && token[0] == '#' && token[1] == '#') {
      // Token pasting operator in body (not expanded argument); discard any
      // immediately preceding white space, then reopen the last token.
      while (!result.empty() && result[result.size() - 1].IsBlank()) {
        result.pop_back();
      }
      if (!result.empty()) {
        result.ReopenLastToken();
        pasting = true;
      }
    } else if (pasting && token.IsBlank()) {
      // Delete whitespace immediately following ## in the body.
    } else if (bytes == 11 && isVariadic_ &&
        token.ToString() == "__VA_ARGS__") {
      for (size_t k{argumentCount_}; k < args.size(); ++k) {
        if (k > argumentCount_) {
          result.Put(","s, allSources.CompilerInsertionProvenance(','));
        }
        result.Put(args[k]);
      }
    } else if (bytes == 10 && isVariadic_ && token.ToString() == "__VA_OPT__" &&
        j + 2 < tokens && replacement_[j + 1].ToString() == "(" &&
        parenthesesNesting == 0) {
      parenthesesNesting = 1;
      skipping = args.size() == argumentCount_;
      ++j;
    } else {
      if (bytes == 1 && parenthesesNesting > 0 && token[0] == '(') {
        ++parenthesesNesting;
      } else if (bytes == 1 && parenthesesNesting > 0 && token[0] == ')') {
        if (--parenthesesNesting == 0) {
          skipping = false;
          continue;
        }
      }
      result.Put(replacement_, j);
    }
  }
  return result;
}

static std::string FormatTime(const std::time_t &now, const char *format) {
  char buffer[16];
  return {buffer,
      std::strftime(buffer, sizeof buffer, format, std::localtime(&now))};
}

Preprocessor::Preprocessor(AllSources *allSources) : allSources_{allSources} {
  // Capture current local date & time once now to avoid having the values
  // of __DATE__ or __TIME__ change during compilation.
  std::time_t now;
  std::time(&now);
  definitions_.emplace(SaveTokenAsName("__DATE__"s),  // e.g., "Jun 16 1904"
      Definition{FormatTime(now, "\"%h %e %Y\""), allSources});
  definitions_.emplace(SaveTokenAsName("__TIME__"s),  // e.g., "23:59:60"
      Definition{FormatTime(now, "\"%T\""), allSources});
  // The values of these predefined macros depend on their invocation sites.
  definitions_.emplace(
      SaveTokenAsName("__FILE__"s), Definition{"__FILE__"s, allSources});
  definitions_.emplace(
      SaveTokenAsName("__LINE__"s), Definition{"__LINE__"s, allSources});
}

bool Preprocessor::MacroReplacement(const TokenSequence &input,
    const Prescanner &prescanner, TokenSequence *result) {
  // Do quick scan for any use of a defined name.
  size_t tokens{input.size()};
  size_t j;
  for (j = 0; j < tokens; ++j) {
    size_t bytes{input[j].size()};
    if (bytes > 0 && IsIdentifierFirstCharacter(input[j][0]) &&
        IsNameDefined(input[j])) {
      break;
    }
  }
  if (j == tokens) {
    return false;  // nothing appeared that could be replaced
  }
  result->Put(input, 0, j);
  for (; j < tokens; ++j) {
    const CharPointerWithLength &token{input[j]};
    if (token.IsBlank() || !IsIdentifierFirstCharacter(token[0])) {
      result->Put(input, j);
      continue;
    }
    auto it = definitions_.find(token);
    if (it == definitions_.end()) {
      result->Put(input, j);
      continue;
    }
    Definition &def{it->second};
    if (def.isDisabled()) {
      result->Put(input, j);
      continue;
    }
    if (!def.isFunctionLike()) {
      if (def.isPredefined()) {
        std::string name{def.replacement()[0].ToString()};
        std::string repl;
        if (name == "__FILE__") {
          repl = "\""s +
              allSources_->GetPath(prescanner.GetCurrentProvenance()) + '"';
        } else if (name == "__LINE__") {
          std::stringstream ss;
          ss << allSources_->GetLineNumber(prescanner.GetCurrentProvenance());
          repl = ss.str();
        }
        if (!repl.empty()) {
          ProvenanceRange insert{allSources_->AddCompilerInsertion(repl)};
          ProvenanceRange call{allSources_->AddMacroCall(
              insert, input.GetTokenProvenanceRange(j), repl)};
          result->Put(repl, call.LocalOffsetToProvenance(0));
          continue;
        }
      }
      def.set_isDisabled(true);
      TokenSequence replaced{ReplaceMacros(def.replacement(), prescanner)};
      def.set_isDisabled(false);
      if (!replaced.empty()) {
        ProvenanceRange from{def.replacement().GetProvenanceRange()};
        ProvenanceRange use{input.GetTokenProvenanceRange(j)};
        ProvenanceRange newRange{
            allSources_->AddMacroCall(from, use, replaced.ToString())};
        result->Put(replaced, newRange);
      }
      continue;
    }
    // Possible function-like macro call.  Skip spaces and newlines to see
    // whether '(' is next.
    size_t k{j};
    bool leftParen{false};
    while (++k < tokens) {
      const CharPointerWithLength &lookAhead{input[k]};
      if (!lookAhead.IsBlank() && lookAhead[0] != '\n') {
        leftParen = lookAhead[0] == '(' && lookAhead.size() == 1;
        break;
      }
    }
    if (!leftParen) {
      result->Put(input, j);
      continue;
    }
    std::vector<size_t> argStart{++k};
    for (int nesting{0}; k < tokens; ++k) {
      if (input[k].size() == 1) {
        char ch{input[k][0]};
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
    if (k >= tokens || argStart.size() < def.argumentCount() ||
        (argStart.size() > def.argumentCount() && !def.isVariadic())) {
      result->Put(input, j);
      continue;
    }
    std::vector<TokenSequence> args;
    for (size_t n{0}; n < argStart.size(); ++n) {
      size_t at{argStart[n]};
      size_t count{(n + 1 == argStart.size() ? k : argStart[n + 1] - 1) - at};
      args.emplace_back(TokenSequence(input, at, count));
    }
    def.set_isDisabled(true);
    TokenSequence replaced{
        ReplaceMacros(def.Apply(args, *allSources_), prescanner)};
    def.set_isDisabled(false);
    if (!replaced.empty()) {
      ProvenanceRange from{def.replacement().GetProvenanceRange()};
      ProvenanceRange use{input.GetIntervalProvenanceRange(j, k - j)};
      ProvenanceRange newRange{
          allSources_->AddMacroCall(from, use, replaced.ToString())};
      result->Put(replaced, newRange);
    }
    j = k;  // advance to the terminal ')'
  }
  return true;
}

TokenSequence Preprocessor::ReplaceMacros(
    const TokenSequence &tokens, const Prescanner &prescanner) {
  TokenSequence repl;
  return MacroReplacement(tokens, prescanner, &repl) ? repl : tokens;
}

static size_t SkipBlanks(
    const TokenSequence &tokens, size_t at, size_t lastToken) {
  for (; at < lastToken; ++at) {
    if (!tokens[at].IsBlank()) {
      break;
    }
  }
  return std::min(at, lastToken);
}

static TokenSequence StripBlanks(
    const TokenSequence &token, size_t first, size_t tokens) {
  TokenSequence noBlanks;
  for (size_t j{SkipBlanks(token, first, tokens)}; j < tokens;
       j = SkipBlanks(token, j + 1, tokens)) {
    noBlanks.Put(token, j);
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
  if (j == tokens || line[j].ToString() != "#") {
    *rest = tokens;
    return {};
  }
  j = SkipBlanks(line, j + 1, tokens);
  if (j == tokens) {
    *rest = tokens;
    return {};
  }
  *rest = SkipBlanks(line, j + 1, tokens);
  return ConvertToLowerCase(line[j].ToString());
}

bool Preprocessor::Directive(const TokenSequence &dir, Prescanner *prescanner) {
  size_t tokens{dir.size()};
  size_t j{SkipBlanks(dir, 0, tokens)};
  if (j == tokens) {
    return true;
  }
  if (dir[j].ToString() != "#") {
    prescanner->Complain("missing '#'");
    return false;
  }
  j = SkipBlanks(dir, j + 1, tokens);
  if (j == tokens) {
    return true;
  }
  if (isdigit(dir[j][0]) || dir[j][0] == '"') {
    return true;  // TODO: treat as #line
  }
  std::string dirName{ConvertToLowerCase(dir[j].ToString())};
  j = SkipBlanks(dir, j + 1, tokens);
  CharPointerWithLength nameToken;
  if (j < tokens && IsIdentifierFirstCharacter(dir[j][0])) {
    nameToken = dir[j];
  }
  if (dirName == "line") {
    // TODO: implement #line
    return true;
  }
  if (dirName == "define") {
    if (nameToken.empty()) {
      prescanner->Complain("#define: missing or invalid name");
      return false;
    }
    nameToken = SaveTokenAsName(nameToken);
    definitions_.erase(nameToken);
    if (++j < tokens && dir[j].size() == 1 && dir[j][0] == '(') {
      j = SkipBlanks(dir, j + 1, tokens);
      std::vector<std::string> argName;
      bool isVariadic{false};
      if (dir[j].ToString() != ")") {
        while (true) {
          std::string an{dir[j].ToString()};
          if (an == "...") {
            isVariadic = true;
          } else {
            if (an.empty() || !IsIdentifierFirstCharacter(an[0])) {
              prescanner->Complain("#define: missing or invalid argument name");
              return false;
            }
            argName.push_back(an);
          }
          j = SkipBlanks(dir, j + 1, tokens);
          if (j == tokens) {
            prescanner->Complain("#define: malformed argument list");
            return false;
          }
          std::string punc{dir[j].ToString()};
          if (punc == ")") {
            break;
          }
          if (punc != ",") {
            prescanner->Complain("#define: malformed argument list");
            return false;
          }
          j = SkipBlanks(dir, j + 1, tokens);
          if (j == tokens || isVariadic) {
            prescanner->Complain("#define: malformed argument list");
            return false;
          }
        }
        if (std::set<std::string>(argName.begin(), argName.end()).size() !=
            argName.size()) {
          prescanner->Complain("#define: argument names are not distinct");
          return false;
        }
      }
      j = SkipBlanks(dir, j + 1, tokens);
      definitions_.emplace(std::make_pair(
          nameToken, Definition{argName, dir, j, tokens - j, isVariadic}));
    } else {
      j = SkipBlanks(dir, j, tokens);
      definitions_.emplace(
          std::make_pair(nameToken, Definition{dir, j, tokens - j}));
    }
    return true;
  }
  if (dirName == "undef") {
    if (nameToken.empty()) {
      prescanner->Complain("# missing or invalid name");
      return false;
    }
    j = SkipBlanks(dir, j + 1, tokens);
    if (j != tokens) {
      prescanner->Complain("#undef: excess tokens at end of directive");
      return false;
    }
    definitions_.erase(nameToken);
    return true;
  }
  if (dirName == "ifdef" || dirName == "ifndef") {
    if (nameToken.empty()) {
      prescanner->Complain("#"s + dirName + ": missing name");
      return false;
    }
    j = SkipBlanks(dir, j + 1, tokens);
    if (j != tokens) {
      prescanner->Complain(
          "#"s + dirName + ": excess tokens at end of directive");
      return false;
    }
    if (IsNameDefined(nameToken) == (dirName == "ifdef")) {
      ifStack_.push(CanDeadElseAppear::Yes);
      return true;
    }
    return SkipDisabledConditionalCode(dirName, IsElseActive::Yes, prescanner);
  }
  if (dirName == "if") {
    if (IsIfPredicateTrue(dir, j, tokens - j, prescanner)) {
      ifStack_.push(CanDeadElseAppear::Yes);
      return true;
    }
    return SkipDisabledConditionalCode(dirName, IsElseActive::Yes, prescanner);
  }
  if (dirName == "else") {
    if (j != tokens) {
      prescanner->Complain("#else: excess tokens at end of directive");
      return false;
    }
    if (ifStack_.empty()) {
      prescanner->Complain("#else: not nested within #if, #ifdef, or #ifndef");
      return false;
    }
    if (ifStack_.top() != CanDeadElseAppear::Yes) {
      prescanner->Complain(
          "#else: already appeared within this #if, #ifdef, or #ifndef");
      return false;
    }
    ifStack_.pop();
    return SkipDisabledConditionalCode("else", IsElseActive::No, prescanner);
  }
  if (dirName == "elif") {
    if (ifStack_.empty()) {
      prescanner->Complain("#elif: not nested within #if, #ifdef, or #ifndef");
      return false;
    }
    if (ifStack_.top() != CanDeadElseAppear::Yes) {
      prescanner->Complain("#elif: #else previously appeared within this "
                           "#if, #ifdef, or #ifndef");
      return false;
    }
    ifStack_.pop();
    return SkipDisabledConditionalCode("elif", IsElseActive::No, prescanner);
  }
  if (dirName == "endif") {
    if (j != tokens) {
      prescanner->Complain("#endif: excess tokens at end of directive");
      return false;
    }
    if (ifStack_.empty()) {
      prescanner->Complain("#endif: no #if, #ifdef, or #ifndef");
      return false;
    }
    ifStack_.pop();
    return true;
  }
  if (dirName == "error" || dirName == "warning") {
    prescanner->Complain(dir.ToString());
    return dirName != "error";
  }
  if (dirName == "include") {
    if (j == tokens) {
      prescanner->Complain("#include: missing name of file to include");
      return false;
    }
    std::string include;
    if (dir[j].ToString() == "<") {
      if (dir[tokens - 1].ToString() != ">") {
        prescanner->Complain("#include: expected '>' at end of directive");
        return false;
      }
      TokenSequence braced{dir, j + 1, tokens - j - 2};
      include = ReplaceMacros(braced, *prescanner).ToString();
    } else if (j + 1 == tokens &&
        (include = dir[j].ToString()).substr(0, 1) == "\"" &&
        include.substr(include.size() - 1, 1) == "\"") {
      include = include.substr(1, include.size() - 2);
    } else {
      prescanner->Complain("#include: expected name of file to include");
      return false;
    }
    if (include.empty()) {
      prescanner->Complain("#include: empty include file name");
      return false;
    }
    std::stringstream error;
    const SourceFile *included{allSources_->Open(include, &error)};
    if (included == nullptr) {
      prescanner->Complain(error.str());
      return false;
    }
    ProvenanceRange fileRange{
        allSources_->AddIncludedFile(*included, dir.GetProvenanceRange())};
    return Prescanner{*prescanner}.Prescan(fileRange);
  }
  prescanner->Complain("#"s + dirName + ": unknown or unimplemented directive");
  return false;
}

CharPointerWithLength Preprocessor::SaveTokenAsName(
    const CharPointerWithLength &t) {
  names_.push_back(t.ToString());
  return {names_.back().data(), names_.back().size()};
}

bool Preprocessor::IsNameDefined(const CharPointerWithLength &token) {
  return definitions_.find(token) != definitions_.end();
}

bool Preprocessor::SkipDisabledConditionalCode(const std::string &dirName,
    IsElseActive isElseActive, Prescanner *prescanner) {
  int nesting{0};
  while (std::optional<TokenSequence> line{prescanner->NextTokenizedLine()}) {
    size_t rest{0};
    std::string dn{GetDirectiveName(*line, &rest)};
    if (dn == "ifdef" || dn == "ifndef" || dn == "if") {
      ++nesting;
    } else if (dn == "endif") {
      if (nesting-- == 0) {
        return true;
      }
    } else if (isElseActive == IsElseActive::Yes && nesting == 0) {
      if (dn == "else") {
        ifStack_.push(CanDeadElseAppear::No);
        return true;
      }
      if (dn == "elif" &&
          IsIfPredicateTrue(*line, rest, line->size() - rest, prescanner)) {
        ifStack_.push(CanDeadElseAppear::Yes);
        return true;
      }
    }
  }
  prescanner->Complain("#"s + dirName + ": missing #endif");
  return false;
}

// Precedence level codes used here to accommodate mixed Fortran and C:
// 15: parentheses and constants, logical !, bitwise ~
// 14: unary + and -
// 13: **
// 12: *, /, % (modulus)
// 11: + and -
// 10: << and >>
//  9: bitwise &
//  8: bitwise ^
//  7: bitwise |
//  6: relations (.EQ., ==, &c.)
//  5: .NOT.
//  4: .AND., &&
//  3: .OR., ||
//  2: .EQV. and .NEQV. / .XOR.
//  1: ? :
//  0: ,
static std::int64_t ExpressionValue(const TokenSequence &token,
    int minimumPrecedence, size_t *atToken, std::string *errors) {
  enum Operator {
    PARENS,
    CONST,
    NOTZERO,  // !
    COMPLEMENT,  // ~
    UPLUS,
    UMINUS,
    POWER,
    TIMES,
    DIVIDE,
    MODULUS,
    ADD,
    SUBTRACT,
    LEFTSHIFT,
    RIGHTSHIFT,
    BITAND,
    BITXOR,
    BITOR,
    LT,
    LE,
    EQ,
    NE,
    GE,
    GT,
    NOT,
    AND,
    OR,
    EQV,
    NEQV,
    SELECT,
    COMMA
  };
  static const int precedence[]{
      15, 15, 15, 15,  // (), 6, !, ~
      14, 14,  // unary +, -
      13, 12, 12, 12, 11, 11, 10, 10,  // **, *, /, %, +, -, <<, >>
      9, 8, 7,  // &, ^, |
      6, 6, 6, 6, 6, 6,  // relations
      5, 4, 3, 2, 2,  // .NOT., .AND., .OR., .EQV., .NEQV.
      1, 0  // ?: and ,
  };
  static const int operandPrecedence[]{0, -1, 15, 15, 15, 15, 13, 12, 12, 12,
      11, 11, 11, 11, 9, 8, 7, 7, 7, 7, 7, 7, 7, 6, 4, 3, 3, 3, 1, 0};

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
    opNameMap["?"] = SELECT;
    opNameMap[","] = COMMA;
  }

  size_t tokens{token.size()};
  if (*atToken >= tokens) {
    *errors = "incomplete expression";
    return 0;
  }
  std::string t{token[*atToken].ToString()};
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
      ConvertToLowerCase(token[*atToken + 1].ToString()) == "not" &&
      token[*atToken + 2].ToString() == ".") {
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
      if (*atToken < tokens && token[*atToken].ToString() == ")") {
        ++*atToken;
      } else if (errors->empty()) {
        *errors = "')' missing from expression";
      }
      break;
    case NOTZERO: left = !left; break;
    case COMPLEMENT: left = ~left; break;
    case UPLUS: break;
    case UMINUS: left = -left; break;
    case NOT: left = -!left; break;
    default: CRASH_NO_CASE;
    }
  }
  if (!errors->empty() || *atToken >= tokens) {
    return left;
  }

  // Parse and evaluate a binary operator and its second operand, if present.
  int advance{1};
  t = token[*atToken].ToString();
  if (t == "." && *atToken + 2 < tokens &&
      token[*atToken + 2].ToString() == ".") {
    t += ConvertToLowerCase(token[*atToken + 1].ToString()) + '.';
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
  std::int64_t right{
      ExpressionValue(token, operandPrecedence[op], atToken, errors)};
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
    {
      std::int64_t power{1};
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
    if ((left < 0) == (right < 0) && (left < 0) != (left + right < 0)) {
      *errors = "overflow in addition";
    }
    return left + right;
  case SUBTRACT:
    if ((left < 0) != (right < 0) && (left < 0) == (left - right < 0)) {
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
  case AND: return left & right;
  case BITXOR: return left ^ right;
  case BITOR:
  case OR: return left | right;
  case LT: return -(left < right);
  case LE: return -(left <= right);
  case EQ: return -(left == right);
  case NE: return -(left != right);
  case GE: return -(left >= right);
  case GT: return -(left > right);
  case EQV: return -(!left == !right);
  case NEQV: return -(!left != !right);
  case SELECT:
    if (*atToken >= tokens || token[*atToken].ToString() != ":") {
      *errors = "':' required in selection expression";
      return left;
    } else {
      ++*atToken;
      std::int64_t third{
          ExpressionValue(token, operandPrecedence[op], atToken, errors)};
      return left != 0 ? right : third;
    }
  case COMMA: return right;
  default: CRASH_NO_CASE;
  }
  return 0;  // silence compiler warning
}

bool Preprocessor::IsIfPredicateTrue(const TokenSequence &expr, size_t first,
    size_t exprTokens, Prescanner *prescanner) {
  TokenSequence expr1{StripBlanks(expr, first, first + exprTokens)};
  TokenSequence expr2;
  for (size_t j{0}; j < expr1.size(); ++j) {
    if (ConvertToLowerCase(expr1[j].ToString()) == "defined") {
      CharPointerWithLength name;
      if (j + 3 < expr1.size() && expr1[j + 1].ToString() == "(" &&
          expr1[j + 3].ToString() == ")") {
        name = expr1[j + 2];
        j += 3;
      } else if (j + 1 < expr1.size() &&
          IsIdentifierFirstCharacter(expr1[j + 1])) {
        name = expr1[j++];
      }
      if (!name.empty()) {
        char truth{IsNameDefined(name) ? '1' : '0'};
        expr2.Put(&truth, 1, allSources_->CompilerInsertionProvenance(truth));
        continue;
      }
    }
    expr2.Put(expr1, j);
  }
  TokenSequence expr3{ReplaceMacros(expr2, *prescanner)};
  TokenSequence expr4{StripBlanks(expr3, 0, expr3.size())};
  size_t atToken{0};
  std::string error;
  bool result{ExpressionValue(expr4, 0, &atToken, &error) != 0};
  if (!error.empty()) {
    prescanner->Complain(error);
  } else if (atToken < expr4.size()) {
    prescanner->Complain(atToken == 0 ? "could not parse any expression"
                                      : "excess characters after expression");
  }
  return result;
}
}  // namespace parser
}  // namespace Fortran
