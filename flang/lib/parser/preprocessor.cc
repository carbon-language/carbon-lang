#include "preprocessor.h"
#include "characters.h"
#include "idioms.h"
#include "message.h"
#include "prescan.h"
#include <algorithm>
#include <cinttypes>
#include <ctime>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <utility>

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
    replacement_{
        predefined, sources->AddCompilerInsertion(predefined).start()} {}

bool Definition::set_isDisabled(bool disable) {
  bool was{isDisabled_};
  isDisabled_ = disable;
  return was;
}

static bool IsLegalIdentifierStart(const ContiguousChars &cpl) {
  return cpl.size() > 0 && IsLegalIdentifierStart(cpl[0]);
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
    ContiguousChars tok{token[firstToken + j]};
    if (IsLegalIdentifierStart(tok)) {
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

static size_t AfterLastNonBlank(const TokenSequence &tokens) {
  for (size_t j{tokens.size()}; j > 0; --j) {
    if (!tokens[j - 1].IsBlank()) {
      return j;
    }
  }
  return 0;
}

static TokenSequence Stringify(
    const TokenSequence &tokens, AllSources *allSources) {
  TokenSequence result;
  Provenance quoteProvenance{allSources->CompilerInsertionProvenance('"')};
  result.PutNextTokenChar('"', quoteProvenance);
  for (size_t j{0}; j < tokens.size(); ++j) {
    const ContiguousChars &token{tokens[j]};
    size_t bytes{token.size()};
    for (size_t k{0}; k < bytes; ++k) {
      char ch{token[k]};
      Provenance from{tokens.GetTokenProvenance(j, k)};
      if (ch == '"' || ch == '\\') {
        result.PutNextTokenChar(ch, from);
      }
      result.PutNextTokenChar(ch, from);
    }
  }
  result.PutNextTokenChar('"', quoteProvenance);
  result.CloseToken();
  return result;
}

TokenSequence Definition::Apply(
    const std::vector<TokenSequence> &args, AllSources *allSources) {
  TokenSequence result;
  bool pasting{false};
  bool skipping{false};
  int parenthesesNesting{0};
  size_t tokens{replacement_.size()};
  for (size_t j{0}; j < tokens; ++j) {
    const ContiguousChars &token{replacement_[j]};
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
      size_t afterLastNonBlank{AfterLastNonBlank(result)};
      if (afterLastNonBlank > 0 &&
          result[afterLastNonBlank - 1].ToString() == "#") {
        // stringifying
        while (result.size() >= afterLastNonBlank) {
          result.pop_back();
        }
        result.Put(Stringify(args[index], allSources));
      } else {
        size_t argTokens{args[index].size()};
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
        token.ToString() == "__VA_ARGs__") {
      Provenance commaProvenance{allSources->CompilerInsertionProvenance(',')};
      for (size_t k{argumentCount_}; k < args.size(); ++k) {
        if (k > argumentCount_) {
          result.Put(","s, commaProvenance);
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

void Preprocessor::Define(std::string macro, std::string value) {
  definitions_.emplace(SaveTokenAsName(macro), Definition{value, allSources_});
}

void Preprocessor::Undefine(std::string macro) { definitions_.erase(macro); }

bool Preprocessor::MacroReplacement(const TokenSequence &input,
    const Prescanner &prescanner, TokenSequence *result) {
  // Do quick scan for any use of a defined name.
  size_t tokens{input.size()};
  size_t j;
  for (j = 0; j < tokens; ++j) {
    size_t bytes{input[j].size()};
    if (bytes > 0 && IsLegalIdentifierStart(input[j][0]) &&
        IsNameDefined(input[j])) {
      break;
    }
  }
  if (j == tokens) {
    return false;  // contains nothing that would be replaced
  }
  result->Put(input, 0, j);
  for (; j < tokens; ++j) {
    const ContiguousChars &token{input[j]};
    if (token.IsBlank() || !IsLegalIdentifierStart(token[0])) {
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
          result->Put(repl, call.start());
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
      const ContiguousChars &lookAhead{input[k]};
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
        ReplaceMacros(def.Apply(args, allSources_), prescanner)};
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

void Preprocessor::Directive(const TokenSequence &dir, Prescanner *prescanner) {
  size_t tokens{dir.size()};
  size_t j{SkipBlanks(dir, 0, tokens)};
  if (j == tokens) {
    return;
  }
  if (dir[j].ToString() != "#") {
    prescanner->Error("missing '#'"_en_US);
    return;
  }
  j = SkipBlanks(dir, j + 1, tokens);
  if (j == tokens) {
    return;
  }
  if (IsDecimalDigit(dir[j][0]) || dir[j][0] == '"') {
    return;  // TODO: treat as #line
  }
  std::string dirName{ToLowerCaseLetters(dir[j].ToString())};
  j = SkipBlanks(dir, j + 1, tokens);
  ContiguousChars nameToken;
  if (j < tokens && IsLegalIdentifierStart(dir[j][0])) {
    nameToken = dir[j];
  }
  if (dirName == "line") {
    // TODO: implement #line
  } else if (dirName == "define") {
    if (nameToken.empty()) {
      prescanner->Error("#define: missing or invalid name"_en_US);
      return;
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
            if (an.empty() || !IsLegalIdentifierStart(an[0])) {
              prescanner->Error(
                  "#define: missing or invalid argument name"_en_US);
              return;
            }
            argName.push_back(an);
          }
          j = SkipBlanks(dir, j + 1, tokens);
          if (j == tokens) {
            prescanner->Error("#define: malformed argument list"_en_US);
            return;
          }
          std::string punc{dir[j].ToString()};
          if (punc == ")") {
            break;
          }
          if (isVariadic || punc != ",") {
            prescanner->Error("#define: malformed argument list"_en_US);
            return;
          }
          j = SkipBlanks(dir, j + 1, tokens);
          if (j == tokens) {
            prescanner->Error("#define: malformed argument list"_en_US);
            return;
          }
        }
        if (std::set<std::string>(argName.begin(), argName.end()).size() !=
            argName.size()) {
          prescanner->Error("#define: argument names are not distinct"_en_US);
          return;
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
  } else if (dirName == "undef") {
    if (nameToken.empty()) {
      prescanner->Error("# missing or invalid name"_en_US);
    } else {
      j = SkipBlanks(dir, j + 1, tokens);
      if (j != tokens) {
        prescanner->Error("#undef: excess tokens at end of directive"_en_US);
      } else {
        definitions_.erase(nameToken);
      }
    }
  } else if (dirName == "ifdef" || dirName == "ifndef") {
    if (nameToken.empty()) {
      prescanner->Error(
          MessageFormattedText("#%s: missing name"_en_US, dirName.data()));
      return;
    }
    j = SkipBlanks(dir, j + 1, tokens);
    if (j != tokens) {
      prescanner->Error(MessageFormattedText(
          "#%s: excess tokens at end of directive"_en_US, dirName.data()));
    } else if (IsNameDefined(nameToken) == (dirName == "ifdef")) {
      ifStack_.push(CanDeadElseAppear::Yes);
    } else {
      SkipDisabledConditionalCode(dirName, IsElseActive::Yes, prescanner);
    }
  } else if (dirName == "if") {
    if (IsIfPredicateTrue(dir, j, tokens - j, prescanner)) {
      ifStack_.push(CanDeadElseAppear::Yes);
    } else {
      SkipDisabledConditionalCode(dirName, IsElseActive::Yes, prescanner);
    }
  } else if (dirName == "else") {
    if (j != tokens) {
      prescanner->Error("#else: excess tokens at end of directive"_en_US);
    } else if (ifStack_.empty()) {
      prescanner->Error(
          "#else: not nested within #if, #ifdef, or #ifndef"_en_US);
    } else if (ifStack_.top() != CanDeadElseAppear::Yes) {
      prescanner->Error(
          "#else: already appeared within this #if, #ifdef, or #ifndef"_en_US);
    } else {
      ifStack_.pop();
      SkipDisabledConditionalCode("else", IsElseActive::No, prescanner);
    }
  } else if (dirName == "elif") {
    if (ifStack_.empty()) {
      prescanner->Error(
          "#elif: not nested within #if, #ifdef, or #ifndef"_en_US);
    } else if (ifStack_.top() != CanDeadElseAppear::Yes) {
      prescanner->Error("#elif: #else previously appeared within this "
                        "#if, #ifdef, or #ifndef"_en_US);
    } else {
      ifStack_.pop();
      SkipDisabledConditionalCode("elif", IsElseActive::No, prescanner);
    }
  } else if (dirName == "endif") {
    if (j != tokens) {
      prescanner->Error("#endif: excess tokens at end of directive"_en_US);
    } else if (ifStack_.empty()) {
      prescanner->Error("#endif: no #if, #ifdef, or #ifndef"_en_US);
    } else {
      ifStack_.pop();
    }
  } else if (dirName == "error") {
    prescanner->Error(
        MessageFormattedText("#error: %s"_en_US, dir.ToString().data()));
  } else if (dirName == "warning") {
    prescanner->Complain(
        MessageFormattedText("#warning: %s"_en_US, dir.ToString().data()));
  } else if (dirName == "include") {
    if (j == tokens) {
      prescanner->Error("#include: missing name of file to include"_en_US);
      return;
    }
    std::string include;
    if (dir[j].ToString() == "<") {
      if (dir[tokens - 1].ToString() != ">") {
        prescanner->Error("#include: expected '>' at end of directive"_en_US);
        return;
      }
      TokenSequence braced{dir, j + 1, tokens - j - 2};
      include = ReplaceMacros(braced, *prescanner).ToString();
    } else if (j + 1 == tokens &&
        (include = dir[j].ToString()).substr(0, 1) == "\"" &&
        include.substr(include.size() - 1, 1) == "\"") {
      include = include.substr(1, include.size() - 2);
    } else {
      prescanner->Error("#include: expected name of file to include"_en_US);
      return;
    }
    if (include.empty()) {
      prescanner->Error("#include: empty include file name"_en_US);
      return;
    }
    std::stringstream error;
    const SourceFile *included{allSources_->Open(include, &error)};
    if (included == nullptr) {
      prescanner->Error(
          MessageFormattedText("#include: %s"_en_US, error.str().data()));
      return;
    }
    ProvenanceRange fileRange{
        allSources_->AddIncludedFile(*included, dir.GetProvenanceRange())};
    if (!Prescanner{*prescanner}.Prescan(fileRange)) {
      prescanner->set_anyFatalErrors();
    }
  } else {
    prescanner->Error(MessageFormattedText(
        "#%s: unknown or unimplemented directive"_en_US, dirName.data()));
  }
}

ContiguousChars Preprocessor::SaveTokenAsName(const ContiguousChars &t) {
  names_.push_back(t.ToString());
  return {names_.back().data(), names_.back().size()};
}

bool Preprocessor::IsNameDefined(const ContiguousChars &token) {
  return definitions_.find(token) != definitions_.end();
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
  return ToLowerCaseLetters(line[j].ToString());
}

void Preprocessor::SkipDisabledConditionalCode(const std::string &dirName,
    IsElseActive isElseActive, Prescanner *prescanner) {
  int nesting{0};
  while (
      std::optional<TokenSequence> line{prescanner->NextTokenizedLine(false)}) {
    size_t rest{0};
    std::string dn{GetDirectiveName(*line, &rest)};
    if (dn == "ifdef" || dn == "ifndef" || dn == "if") {
      ++nesting;
    } else if (dn == "endif") {
      if (nesting-- == 0) {
        return;
      }
    } else if (isElseActive == IsElseActive::Yes && nesting == 0) {
      if (dn == "else") {
        ifStack_.push(CanDeadElseAppear::No);
        return;
      }
      if (dn == "elif" &&
          IsIfPredicateTrue(*line, rest, line->size() - rest, prescanner)) {
        ifStack_.push(CanDeadElseAppear::Yes);
        return;
      }
    }
  }
  prescanner->Error(
      MessageFormattedText("#%s: missing #endif"_en_US, dirName.data()));
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
    int minimumPrecedence, size_t *atToken, MessageFixedText *error) {
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
    *error = "incomplete expression"_en_US;
    return 0;
  }
  std::string t{token[*atToken].ToString()};
  enum Operator op;

  // Parse and evaluate a primary or a unary operator and its operand.
  std::int64_t left{0};
  if (t == "(") {
    op = PARENS;
  } else if (IsDecimalDigit(t[0])) {
    op = CONST;
    size_t consumed{0};
    left = std::stoll(t, &consumed);
    if (consumed < t.size()) {
      *error = "uninterpretable numeric constant '"_en_US;
    }
  } else if (IsLegalIdentifierStart(t[0])) {
    // undefined macro name -> zero
    // TODO: BOZ constants?
    op = CONST;
  } else if (t == "+") {
    op = UPLUS;
  } else if (t == "-") {
    op = UMINUS;
  } else if (t == "." && *atToken + 2 < tokens &&
      ToLowerCaseLetters(token[*atToken + 1].ToString()) == "not" &&
      token[*atToken + 2].ToString() == ".") {
    op = NOT;
    *atToken += 2;
  } else {
    auto it = opNameMap.find(t);
    if (it != opNameMap.end()) {
      op = it->second;
    } else {
      *error = "operand expected in expression"_en_US;
      return 0;
    }
  }
  if (precedence[op] < minimumPrecedence && error->empty()) {
    *error = "operator precedence error"_en_US;
  }
  ++*atToken;
  if (op != CONST && error->empty()) {
    left = ExpressionValue(token, operandPrecedence[op], atToken, error);
    switch (op) {
    case PARENS:
      if (*atToken < tokens && token[*atToken].ToString() == ")") {
        ++*atToken;
      } else if (error->empty()) {
        *error = "')' missing from expression"_en_US;
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
  if (!error->empty() || *atToken >= tokens) {
    return left;
  }

  // Parse and evaluate a binary operator and its second operand, if present.
  int advance{1};
  t = token[*atToken].ToString();
  if (t == "." && *atToken + 2 < tokens &&
      token[*atToken + 2].ToString() == ".") {
    t += ToLowerCaseLetters(token[*atToken + 1].ToString()) + '.';
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
      ExpressionValue(token, operandPrecedence[op], atToken, error)};
  switch (op) {
  case POWER:
    if (left == 0 && right < 0) {
      *error = "0 ** negative power"_en_US;
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
          *error = "overflow in exponentation"_en_US;
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
      *error = "overflow in multiplication"_en_US;
    }
    return left * right;
  case DIVIDE:
    if (right == 0) {
      *error = "division by zero"_en_US;
      return 0;
    }
    return left / right;
  case MODULUS:
    if (right == 0) {
      *error = "modulus by zero"_en_US;
      return 0;
    }
    return left % right;
  case ADD:
    if ((left < 0) == (right < 0) && (left < 0) != (left + right < 0)) {
      *error = "overflow in addition"_en_US;
    }
    return left + right;
  case SUBTRACT:
    if ((left < 0) != (right < 0) && (left < 0) == (left - right < 0)) {
      *error = "overflow in subtraction"_en_US;
    }
    return left - right;
  case LEFTSHIFT:
    if (right < 0 || right > 64) {
      *error = "bad left shift count"_en_US;
    }
    return right >= 64 ? 0 : left << right;
  case RIGHTSHIFT:
    if (right < 0 || right > 64) {
      *error = "bad right shift count"_en_US;
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
      *error = "':' required in selection expression"_en_US;
      return left;
    } else {
      ++*atToken;
      std::int64_t third{
          ExpressionValue(token, operandPrecedence[op], atToken, error)};
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
    if (ToLowerCaseLetters(expr1[j].ToString()) == "defined") {
      ContiguousChars name;
      if (j + 3 < expr1.size() && expr1[j + 1].ToString() == "(" &&
          expr1[j + 3].ToString() == ")") {
        name = expr1[j + 2];
        j += 3;
      } else if (j + 1 < expr1.size() && IsLegalIdentifierStart(expr1[j + 1])) {
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
  MessageFixedText error;
  bool result{ExpressionValue(expr4, 0, &atToken, &error) != 0};
  if (!error.empty()) {
    prescanner->Error(error);
  } else if (atToken < expr4.size()) {
    prescanner->Error(atToken == 0
            ? "could not parse any expression"_en_US
            : "excess characters after expression"_en_US);
  }
  return result;
}
}  // namespace parser
}  // namespace Fortran
