#include "preprocessor.h"
#include "characters.h"
#include "idioms.h"
#include "message.h"
#include "prescan.h"
#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <ctime>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <utility>

namespace Fortran {
namespace parser {

Definition::Definition(
    const TokenSequence &repl, std::size_t firstToken, std::size_t tokens)
  : replacement_{Tokenize({}, repl, firstToken, tokens)} {}

Definition::Definition(const std::vector<std::string> &argNames,
    const TokenSequence &repl, std::size_t firstToken, std::size_t tokens,
    bool isVariadic)
  : isFunctionLike_{true},
    argumentCount_(argNames.size()), isVariadic_{isVariadic},
    replacement_{Tokenize(argNames, repl, firstToken, tokens)} {}

Definition::Definition(const std::string &predefined, AllSources &sources)
  : isPredefined_{true}, replacement_{predefined,
                             sources.AddCompilerInsertion(predefined).start()} {
}

bool Definition::set_isDisabled(bool disable) {
  bool was{isDisabled_};
  isDisabled_ = disable;
  return was;
}

static bool IsLegalIdentifierStart(const CharBlock &cpl) {
  return cpl.size() > 0 && IsLegalIdentifierStart(cpl[0]);
}

TokenSequence Definition::Tokenize(const std::vector<std::string> &argNames,
    const TokenSequence &token, std::size_t firstToken, std::size_t tokens) {
  std::map<std::string, std::string> args;
  char argIndex{'A'};
  for (const std::string &arg : argNames) {
    CHECK(args.find(arg) == args.end());
    args[arg] = "~"s + argIndex++;
  }
  TokenSequence result;
  for (std::size_t j{0}; j < tokens; ++j) {
    CharBlock tok{token.TokenAt(firstToken + j)};
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

static std::size_t AfterLastNonBlank(const TokenSequence &tokens) {
  for (std::size_t j{tokens.SizeInTokens()}; j > 0; --j) {
    if (!tokens.TokenAt(j - 1).IsBlank()) {
      return j;
    }
  }
  return 0;
}

static TokenSequence Stringify(
    const TokenSequence &tokens, AllSources &allSources) {
  TokenSequence result;
  Provenance quoteProvenance{allSources.CompilerInsertionProvenance('"')};
  result.PutNextTokenChar('"', quoteProvenance);
  for (std::size_t j{0}; j < tokens.SizeInTokens(); ++j) {
    const CharBlock &token{tokens.TokenAt(j)};
    std::size_t bytes{token.size()};
    for (std::size_t k{0}; k < bytes; ++k) {
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
    const std::vector<TokenSequence> &args, AllSources &allSources) {
  TokenSequence result;
  bool pasting{false};
  bool skipping{false};
  int parenthesesNesting{0};
  std::size_t tokens{replacement_.SizeInTokens()};
  for (std::size_t j{0}; j < tokens; ++j) {
    const CharBlock &token{replacement_.TokenAt(j)};
    std::size_t bytes{token.size()};
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
      std::size_t index = token[1] - 'A';
      if (index >= args.size()) {
        continue;
      }
      std::size_t afterLastNonBlank{AfterLastNonBlank(result)};
      if (afterLastNonBlank > 0 &&
          result.TokenAt(afterLastNonBlank - 1).ToString() == "#") {
        // stringifying
        while (result.SizeInTokens() >= afterLastNonBlank) {
          result.pop_back();
        }
        result.Put(Stringify(args[index], allSources));
      } else {
        std::size_t argTokens{args[index].SizeInTokens()};
        for (std::size_t k{0}; k < argTokens; ++k) {
          if (!pasting || !args[index].TokenAt(k).IsBlank()) {
            result.Put(args[index], k);
            pasting = false;
          }
        }
      }
    } else if (bytes == 2 && token[0] == '#' && token[1] == '#') {
      // Token pasting operator in body (not expanded argument); discard any
      // immediately preceding white space, then reopen the last token.
      while (!result.empty() &&
          result.TokenAt(result.SizeInTokens() - 1).IsBlank()) {
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
      Provenance commaProvenance{allSources.CompilerInsertionProvenance(',')};
      for (std::size_t k{argumentCount_}; k < args.size(); ++k) {
        if (k > argumentCount_) {
          result.Put(","s, commaProvenance);
        }
        result.Put(args[k]);
      }
    } else if (bytes == 10 && isVariadic_ && token.ToString() == "__VA_OPT__" &&
        j + 2 < tokens && replacement_.TokenAt(j + 1).ToString() == "(" &&
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

Preprocessor::Preprocessor(AllSources &allSources) : allSources_{allSources} {
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

std::optional<TokenSequence> Preprocessor::MacroReplacement(
    const TokenSequence &input, const Prescanner &prescanner) {
  // Do quick scan for any use of a defined name.
  std::size_t tokens{input.SizeInTokens()};
  std::size_t j;
  for (j = 0; j < tokens; ++j) {
    CharBlock token{input.TokenAt(j)};
    if (!token.empty() && IsLegalIdentifierStart(token[0]) &&
        IsNameDefined(token)) {
      break;
    }
  }
  if (j == tokens) {
    return {};  // input contains nothing that would be replaced
  }
  TokenSequence result{input, 0, j};
  for (; j < tokens; ++j) {
    const CharBlock &token{input.TokenAt(j)};
    if (token.IsBlank() || !IsLegalIdentifierStart(token[0])) {
      result.Put(input, j);
      continue;
    }
    auto it = definitions_.find(token);
    if (it == definitions_.end()) {
      result.Put(input, j);
      continue;
    }
    Definition &def{it->second};
    if (def.isDisabled()) {
      result.Put(input, j);
      continue;
    }
    if (!def.isFunctionLike()) {
      if (def.isPredefined()) {
        std::string name{def.replacement().TokenAt(0).ToString()};
        std::string repl;
        if (name == "__FILE__") {
          repl = "\""s +
              allSources_.GetPath(prescanner.GetCurrentProvenance()) + '"';
        } else if (name == "__LINE__") {
          std::stringstream ss;
          ss << allSources_.GetLineNumber(prescanner.GetCurrentProvenance());
          repl = ss.str();
        }
        if (!repl.empty()) {
          ProvenanceRange insert{allSources_.AddCompilerInsertion(repl)};
          ProvenanceRange call{allSources_.AddMacroCall(
              insert, input.GetTokenProvenanceRange(j), repl)};
          result.Put(repl, call.start());
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
            allSources_.AddMacroCall(from, use, replaced.ToString())};
        result.Put(replaced, newRange);
      }
      continue;
    }
    // Possible function-like macro call.  Skip spaces and newlines to see
    // whether '(' is next.
    std::size_t k{j};
    bool leftParen{false};
    while (++k < tokens) {
      const CharBlock &lookAhead{input.TokenAt(k)};
      if (!lookAhead.IsBlank() && lookAhead[0] != '\n') {
        leftParen = lookAhead[0] == '(' && lookAhead.size() == 1;
        break;
      }
    }
    if (!leftParen) {
      result.Put(input, j);
      continue;
    }
    std::vector<std::size_t> argStart{++k};
    for (int nesting{0}; k < tokens; ++k) {
      CharBlock token{input.TokenAt(k)};
      if (token.size() == 1) {
        char ch{token[0]};
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
      result.Put(input, j);
      continue;
    }
    std::vector<TokenSequence> args;
    for (std::size_t n{0}; n < argStart.size(); ++n) {
      std::size_t at{argStart[n]};
      std::size_t count{
          (n + 1 == argStart.size() ? k : argStart[n + 1] - 1) - at};
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
          allSources_.AddMacroCall(from, use, replaced.ToString())};
      result.Put(replaced, newRange);
    }
    j = k;  // advance to the terminal ')'
  }
  return {result};
}

TokenSequence Preprocessor::ReplaceMacros(
    const TokenSequence &tokens, const Prescanner &prescanner) {
  if (std::optional<TokenSequence> repl{MacroReplacement(tokens, prescanner)}) {
    return std::move(*repl);
  }
  return tokens;
}

static std::size_t SkipBlanks(
    const TokenSequence &tokens, std::size_t at, std::size_t lastToken) {
  for (; at < lastToken; ++at) {
    if (!tokens.TokenAt(at).IsBlank()) {
      break;
    }
  }
  return std::min(at, lastToken);
}

static TokenSequence StripBlanks(
    const TokenSequence &token, std::size_t first, std::size_t tokens) {
  TokenSequence noBlanks;
  for (std::size_t j{SkipBlanks(token, first, tokens)}; j < tokens;
       j = SkipBlanks(token, j + 1, tokens)) {
    noBlanks.Put(token, j);
  }
  return noBlanks;
}

void Preprocessor::Directive(const TokenSequence &dir, Prescanner *prescanner) {
  std::size_t tokens{dir.SizeInTokens()};
  std::size_t j{SkipBlanks(dir, 0, tokens)};
  if (j == tokens) {
    return;
  }
  if (dir.TokenAt(j).ToString() != "#") {
    prescanner->Say("missing '#'"_err_en_US, dir.GetTokenProvenance(j));
    return;
  }
  j = SkipBlanks(dir, j + 1, tokens);
  if (j == tokens) {
    return;
  }
  if (IsDecimalDigit(dir.TokenAt(j)[0]) || dir.TokenAt(j)[0] == '"') {
    return;  // treat like #line, ignore it
  }
  std::size_t dirOffset{j};
  std::string dirName{ToLowerCaseLetters(dir.TokenAt(dirOffset).ToString())};
  j = SkipBlanks(dir, j + 1, tokens);
  CharBlock nameToken;
  if (j < tokens && IsLegalIdentifierStart(dir.TokenAt(j)[0])) {
    nameToken = dir.TokenAt(j);
  }
  if (dirName == "line") {
    // #line is ignored
  } else if (dirName == "define") {
    if (nameToken.empty()) {
      prescanner->Say("#define: missing or invalid name"_err_en_US,
          dir.GetTokenProvenance(j < tokens ? j : tokens - 1));
      return;
    }
    nameToken = SaveTokenAsName(nameToken);
    definitions_.erase(nameToken);
    if (++j < tokens && dir.TokenAt(j).size() == 1 &&
        dir.TokenAt(j)[0] == '(') {
      j = SkipBlanks(dir, j + 1, tokens);
      std::vector<std::string> argName;
      bool isVariadic{false};
      if (dir.TokenAt(j).ToString() != ")") {
        while (true) {
          std::string an{dir.TokenAt(j).ToString()};
          if (an == "...") {
            isVariadic = true;
          } else {
            if (an.empty() || !IsLegalIdentifierStart(an[0])) {
              prescanner->Say(
                  "#define: missing or invalid argument name"_err_en_US,
                  dir.GetTokenProvenance(j));
              return;
            }
            argName.push_back(an);
          }
          j = SkipBlanks(dir, j + 1, tokens);
          if (j == tokens) {
            prescanner->Say("#define: malformed argument list"_err_en_US,
                dir.GetTokenProvenance(tokens - 1));
            return;
          }
          std::string punc{dir.TokenAt(j).ToString()};
          if (punc == ")") {
            break;
          }
          if (isVariadic || punc != ",") {
            prescanner->Say("#define: malformed argument list"_err_en_US,
                dir.GetTokenProvenance(j));
            return;
          }
          j = SkipBlanks(dir, j + 1, tokens);
          if (j == tokens) {
            prescanner->Say("#define: malformed argument list"_err_en_US,
                dir.GetTokenProvenance(tokens - 1));
            return;
          }
        }
        if (std::set<std::string>(argName.begin(), argName.end()).size() !=
            argName.size()) {
          prescanner->Say("#define: argument names are not distinct"_err_en_US,
              dir.GetTokenProvenance(dirOffset));
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
      prescanner->Say("# missing or invalid name"_err_en_US,
          dir.GetTokenProvenance(tokens - 1));
    } else {
      j = SkipBlanks(dir, j + 1, tokens);
      if (j != tokens) {
        prescanner->Say("#undef: excess tokens at end of directive"_err_en_US,
            dir.GetTokenProvenance(j));
      } else {
        definitions_.erase(nameToken);
      }
    }
  } else if (dirName == "ifdef" || dirName == "ifndef") {
    if (nameToken.empty()) {
      prescanner->Say(
          MessageFormattedText("#%s: missing name"_err_en_US, dirName.data()),
          dir.GetTokenProvenance(tokens - 1));
      return;
    }
    j = SkipBlanks(dir, j + 1, tokens);
    if (j != tokens) {
      prescanner->Say(MessageFormattedText(
                          "#%s: excess tokens at end of directive"_err_en_US,
                          dirName.data()),
          dir.GetTokenProvenance(j));
    } else if (IsNameDefined(nameToken) == (dirName == "ifdef")) {
      ifStack_.push(CanDeadElseAppear::Yes);
    } else {
      SkipDisabledConditionalCode(dirName, IsElseActive::Yes, prescanner,
          dir.GetTokenProvenance(dirOffset));
    }
  } else if (dirName == "if") {
    if (IsIfPredicateTrue(dir, j, tokens - j, prescanner)) {
      ifStack_.push(CanDeadElseAppear::Yes);
    } else {
      SkipDisabledConditionalCode(dirName, IsElseActive::Yes, prescanner,
          dir.GetTokenProvenance(dirOffset));
    }
  } else if (dirName == "else") {
    if (j != tokens) {
      prescanner->Say("#else: excess tokens at end of directive"_err_en_US,
          dir.GetTokenProvenance(j));
    } else if (ifStack_.empty()) {
      prescanner->Say(
          "#else: not nested within #if, #ifdef, or #ifndef"_err_en_US,
          dir.GetTokenProvenance(tokens - 1));
    } else if (ifStack_.top() != CanDeadElseAppear::Yes) {
      prescanner->Say(
          "#else: already appeared within this #if, #ifdef, or #ifndef"_err_en_US,
          dir.GetTokenProvenance(tokens - 1));
    } else {
      ifStack_.pop();
      SkipDisabledConditionalCode("else", IsElseActive::No, prescanner,
          dir.GetTokenProvenance(dirOffset));
    }
  } else if (dirName == "elif") {
    if (ifStack_.empty()) {
      prescanner->Say(
          "#elif: not nested within #if, #ifdef, or #ifndef"_err_en_US,
          dir.GetTokenProvenance(tokens - 1));
    } else if (ifStack_.top() != CanDeadElseAppear::Yes) {
      prescanner->Say("#elif: #else previously appeared within this "
                      "#if, #ifdef, or #ifndef"_err_en_US,
          dir.GetTokenProvenance(tokens - 1));
    } else {
      ifStack_.pop();
      SkipDisabledConditionalCode("elif", IsElseActive::No, prescanner,
          dir.GetTokenProvenance(dirOffset));
    }
  } else if (dirName == "endif") {
    if (j != tokens) {
      prescanner->Say("#endif: excess tokens at end of directive"_err_en_US,
          dir.GetTokenProvenance(j));
    } else if (ifStack_.empty()) {
      prescanner->Say("#endif: no #if, #ifdef, or #ifndef"_err_en_US,
          dir.GetTokenProvenance(tokens - 1));
    } else {
      ifStack_.pop();
    }
  } else if (dirName == "error") {
    prescanner->Say(
        MessageFormattedText("#error: %s"_err_en_US, dir.ToString().data()),
        dir.GetTokenProvenance(dirOffset));
  } else if (dirName == "warning") {
    prescanner->Say(
        MessageFormattedText("#warning: %s"_en_US, dir.ToString().data()),
        dir.GetTokenProvenance(dirOffset));
  } else if (dirName == "include") {
    if (j == tokens) {
      prescanner->Say("#include: missing name of file to include"_err_en_US,
          dir.GetTokenProvenance(tokens - 1));
      return;
    }
    std::string include;
    if (dir.TokenAt(j).ToString() == "<") {
      if (dir.TokenAt(tokens - 1).ToString() != ">") {
        prescanner->Say("#include: expected '>' at end of directive"_err_en_US,
            dir.GetTokenProvenance(tokens - 1));
        return;
      }
      TokenSequence braced{dir, j + 1, tokens - j - 2};
      include = ReplaceMacros(braced, *prescanner).ToString();
    } else if (j + 1 == tokens &&
        (include = dir.TokenAt(j).ToString()).substr(0, 1) == "\"" &&
        include.substr(include.size() - 1, 1) == "\"") {
      include = include.substr(1, include.size() - 2);
    } else {
      prescanner->Say("#include: expected name of file to include"_err_en_US,
          dir.GetTokenProvenance(j < tokens ? j : tokens - 1));
      return;
    }
    if (include.empty()) {
      prescanner->Say("#include: empty include file name"_err_en_US,
          dir.GetTokenProvenance(dirOffset));
      return;
    }
    std::stringstream error;
    const SourceFile *included{allSources_.Open(include, &error)};
    if (included == nullptr) {
      prescanner->Say(
          MessageFormattedText("#include: %s"_err_en_US, error.str().data()),
          dir.GetTokenProvenance(dirOffset));
    } else if (included->bytes() > 0) {
      ProvenanceRange fileRange{
          allSources_.AddIncludedFile(*included, dir.GetProvenanceRange())};
      Prescanner{*prescanner}.Prescan(fileRange);
    }
  } else {
    prescanner->Say(MessageFormattedText(
                        "#%s: unknown or unimplemented directive"_err_en_US,
                        dirName.data()),
        dir.GetTokenProvenance(dirOffset));
  }
}

CharBlock Preprocessor::SaveTokenAsName(const CharBlock &t) {
  names_.push_back(t.ToString());
  return {names_.back().data(), names_.back().size()};
}

bool Preprocessor::IsNameDefined(const CharBlock &token) {
  return definitions_.find(token) != definitions_.end();
}

static std::string GetDirectiveName(
    const TokenSequence &line, std::size_t *rest) {
  std::size_t tokens{line.SizeInTokens()};
  std::size_t j{SkipBlanks(line, 0, tokens)};
  if (j == tokens || line.TokenAt(j).ToString() != "#") {
    *rest = tokens;
    return "";
  }
  j = SkipBlanks(line, j + 1, tokens);
  if (j == tokens) {
    *rest = tokens;
    return "";
  }
  *rest = SkipBlanks(line, j + 1, tokens);
  return ToLowerCaseLetters(line.TokenAt(j).ToString());
}

void Preprocessor::SkipDisabledConditionalCode(const std::string &dirName,
    IsElseActive isElseActive, Prescanner *prescanner, Provenance provenance) {
  int nesting{0};
  while (!prescanner->IsAtEnd()) {
    if (!prescanner->IsNextLinePreprocessorDirective()) {
      prescanner->NextLine();
      continue;
    }
    TokenSequence line{prescanner->TokenizePreprocessorDirective()};
    std::size_t rest{0};
    std::string dn{GetDirectiveName(line, &rest)};
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
          IsIfPredicateTrue(
              line, rest, line.SizeInTokens() - rest, prescanner)) {
        ifStack_.push(CanDeadElseAppear::Yes);
        return;
      }
    }
  }
  prescanner->Say(
      MessageFormattedText("#%s: missing #endif"_err_en_US, dirName.data()),
      provenance);
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
    int minimumPrecedence, std::size_t *atToken,
    std::optional<Message> *error) {
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

  std::size_t tokens{token.SizeInTokens()};
  if (*atToken >= tokens) {
    *error = Message{token.GetTokenProvenance(tokens - 1),
        "incomplete expression"_err_en_US};
    return 0;
  }

  // Parse and evaluate a primary or a unary operator and its operand.
  std::size_t opAt{*atToken};
  std::string t{token.TokenAt(opAt).ToString()};
  enum Operator op;
  std::int64_t left{0};
  if (t == "(") {
    op = PARENS;
  } else if (IsDecimalDigit(t[0])) {
    op = CONST;
    std::size_t consumed{0};
    left = std::stoll(t, &consumed, 0 /*base to be detected*/);
    if (consumed < t.size()) {
      *error = Message{token.GetTokenProvenance(opAt),
          MessageFormattedText(
              "uninterpretable numeric constant '%s'"_err_en_US, t.data())};
      return 0;
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
      ToLowerCaseLetters(token.TokenAt(*atToken + 1).ToString()) == "not" &&
      token.TokenAt(*atToken + 2).ToString() == ".") {
    op = NOT;
    *atToken += 2;
  } else {
    auto it = opNameMap.find(t);
    if (it != opNameMap.end()) {
      op = it->second;
    } else {
      *error = Message{token.GetTokenProvenance(tokens - 1),
          "operand expected in expression"_err_en_US};
      return 0;
    }
  }
  if (precedence[op] < minimumPrecedence) {
    *error = Message{
        token.GetTokenProvenance(opAt), "operator precedence error"_err_en_US};
    return 0;
  }
  ++*atToken;
  if (op != CONST) {
    left = ExpressionValue(token, operandPrecedence[op], atToken, error);
    if (error->has_value()) {
      return 0;
    }
    switch (op) {
    case PARENS:
      if (*atToken < tokens && token.TokenAt(*atToken).ToString() == ")") {
        ++*atToken;
      } else {
        *error = Message{token.GetTokenProvenance(tokens - 1),
            "')' missing from expression"_err_en_US};
        return 0;
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
  if (*atToken >= tokens) {
    return left;
  }

  // Parse and evaluate a binary operator and its second operand, if present.
  int advance{1};
  t = token.TokenAt(*atToken).ToString();
  if (t == "." && *atToken + 2 < tokens &&
      token.TokenAt(*atToken + 2).ToString() == ".") {
    t += ToLowerCaseLetters(token.TokenAt(*atToken + 1).ToString()) + '.';
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
  opAt = *atToken;
  *atToken += advance;
  std::int64_t right{
      ExpressionValue(token, operandPrecedence[op], atToken, error)};
  if (error->has_value()) {
    return 0;
  }
  switch (op) {
  case POWER:
    if (left == 0 && right < 0) {
      *error = Message{
          token.GetTokenProvenance(opAt), "0 ** negative power"_err_en_US};
      return 0;
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
          *error = Message{token.GetTokenProvenance(opAt),
              "overflow in exponentation"_err_en_US};
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
      *error = Message{token.GetTokenProvenance(opAt),
          "overflow in multiplication"_err_en_US};
    }
    return left * right;
  case DIVIDE:
    if (right == 0) {
      *error =
          Message{token.GetTokenProvenance(opAt), "division by zero"_err_en_US};
      return 0;
    }
    return left / right;
  case MODULUS:
    if (right == 0) {
      *error =
          Message{token.GetTokenProvenance(opAt), "modulus by zero"_err_en_US};
      return 0;
    }
    return left % right;
  case ADD:
    if ((left < 0) == (right < 0) && (left < 0) != (left + right < 0)) {
      *error = Message{
          token.GetTokenProvenance(opAt), "overflow in addition"_err_en_US};
    }
    return left + right;
  case SUBTRACT:
    if ((left < 0) != (right < 0) && (left < 0) == (left - right < 0)) {
      *error = Message{
          token.GetTokenProvenance(opAt), "overflow in subtraction"_err_en_US};
    }
    return left - right;
  case LEFTSHIFT:
    if (right < 0 || right > 64) {
      *error = Message{
          token.GetTokenProvenance(opAt), "bad left shift count"_err_en_US};
    }
    return right >= 64 ? 0 : left << right;
  case RIGHTSHIFT:
    if (right < 0 || right > 64) {
      *error = Message{
          token.GetTokenProvenance(opAt), "bad right shift count"_err_en_US};
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
    if (*atToken >= tokens || token.TokenAt(*atToken).ToString() != ":") {
      *error = Message{token.GetTokenProvenance(opAt),
          "':' required in selection expression"_err_en_US};
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

bool Preprocessor::IsIfPredicateTrue(const TokenSequence &expr,
    std::size_t first, std::size_t exprTokens, Prescanner *prescanner) {
  TokenSequence expr1{StripBlanks(expr, first, first + exprTokens)};
  TokenSequence expr2;
  for (std::size_t j{0}; j < expr1.SizeInTokens(); ++j) {
    if (ToLowerCaseLetters(expr1.TokenAt(j).ToString()) == "defined") {
      CharBlock name;
      if (j + 3 < expr1.SizeInTokens() &&
          expr1.TokenAt(j + 1).ToString() == "(" &&
          expr1.TokenAt(j + 3).ToString() == ")") {
        name = expr1.TokenAt(j + 2);
        j += 3;
      } else if (j + 1 < expr1.SizeInTokens() &&
          IsLegalIdentifierStart(expr1.TokenAt(j + 1))) {
        name = expr1.TokenAt(j++);
      }
      if (!name.empty()) {
        char truth{IsNameDefined(name) ? '1' : '0'};
        expr2.Put(&truth, 1, allSources_.CompilerInsertionProvenance(truth));
        continue;
      }
    }
    expr2.Put(expr1, j);
  }
  TokenSequence expr3{ReplaceMacros(expr2, *prescanner)};
  TokenSequence expr4{StripBlanks(expr3, 0, expr3.SizeInTokens())};
  std::size_t atToken{0};
  std::optional<Message> error;
  bool result{ExpressionValue(expr4, 0, &atToken, &error) != 0};
  if (error.has_value()) {
    prescanner->Say(std::move(*error));
  } else if (atToken < expr4.SizeInTokens()) {
    prescanner->Say(atToken == 0
            ? "could not parse any expression"_err_en_US
            : "excess characters after expression"_err_en_US,
        expr4.GetTokenProvenance(atToken));
  }
  return result;
}
}  // namespace parser
}  // namespace Fortran
