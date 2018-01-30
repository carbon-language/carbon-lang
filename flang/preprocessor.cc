#include "preprocessor.h"
#include "char-buffer.h"
#include "idioms.h"
#include "prescan.h"
#include <cctype>
#include <map>
#include <memory>
#include <set>
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

bool Definition::set_isDisabled(bool disable) {
  bool was{isDisabled_};
  isDisabled_ = disable;
  return was;
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
    if (bytes > 0 && (*text == '_' || isalpha(*text))) {
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

TokenSequence Definition::Apply(const std::vector<TokenSequence> &args) {
  TokenSequence result;
  bool stringify{false}, pasting{false};
  size_t tokens{replacement_.size()};
  for (size_t j{0}; j < tokens; ++j) {
    size_t bytes{replacement_.GetBytes(j)};
    const char *text{replacement_.GetText(j)};
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
          const char *text{args[index].GetText(k)};
          size_t bytes{args[index].GetBytes(k)};
          if (pasting && (bytes == 0 || *text == ' ' || *text == '\t')) {
          } else {
            result.push_back(text, bytes);
            pasting = false;
          }
        }
      }
    } else if (bytes == 2 && text[0] == '#' && text[1] == '#') {
      // Token pasting operator in body (not expanded argument); discard any
      // immediately preceding white space, then reopen the last token.
      while (!result.empty() &&
             (result.GetBytes(result.size() - 1) == 0 ||
              *result.GetText(result.size() - 1) == ' ' ||
              *result.GetText(result.size() - 1) == '\t')) {
        result.pop_back();
      }
      if (!result.empty()) {
        result.ReopenLastToken();
        pasting = true;
      }
    } else if (pasting && (bytes == 0 || *text == ' ' || *text == '\t')) {
      // Delete whitespace immediately following ## in the body.
    } else {
      stringify = bytes == 1 && *text == '#';
      result.push_back(text, bytes);
      pasting = false;
    }
  }
  return result;
}

bool Preprocessor::MacroReplacement(const TokenSequence &input,
                                    TokenSequence *result) {
  // Do quick scan for any use of a defined name.
  if (definitions_.empty()) {
    return false;
  }
  size_t tokens{input.size()};
  size_t j;
  for (j = 0; j < tokens; ++j) {
    const char *text{input.GetText(j)};
    size_t bytes{input.GetBytes(j)};
    if (bytes > 0 &&
        (*text == '_' || isalpha(*text)) &&
        definitions_.find(CharPointerWithLength{text, bytes}) !=
          definitions_.end()) {
      break;
    }
  }
  if (j == tokens) {
    return false;  // nothing appeared that could be replaced
  }

  for (size_t k{0}; k < j; ++k) {
    result->push_back(input.GetToken(k));
  }
  for (; j < tokens; ++j) {
    size_t bytes{input.GetBytes(j)};
    const char *text{input.GetText(j)};
    if (bytes == 0 || (!isalpha(*text) && *text != '_')) {
      result->push_back(text, bytes);
      continue;
    }
    auto it = definitions_.find(CharPointerWithLength{text, bytes});
    if (it == definitions_.end()) {
      result->push_back(text, bytes);
      continue;
    }
    Definition &def{it->second};
    if (def.isDisabled()) {
      result->push_back(text, bytes);
      continue;
    }
    if (!def.isFunctionLike()) {
      def.set_isDisabled(true);
      TokenSequence repl;
      result->Append(MacroReplacement(def.replacement(), &repl) ? repl
                       : def.replacement());
      def.set_isDisabled(false);
      continue;
    }
    // Possible function-like macro call.  Skip spaces and newlines to see
    // whether '(' is next.
    size_t k{j};
    bool leftParen{false};
    while (++k < tokens) {
      size_t bytes{input.GetBytes(k)};
      const char *text{input.GetText(k)};
      if (bytes > 0 && *text != ' ' && *text != '\n') {
        leftParen = bytes == 1 && *text == '(';
        break;
      }
    }
    if (!leftParen) {
      result->push_back(text, bytes);
      continue;
    }
    std::vector<size_t> argStart{++k};
    for (int nesting{0}; k < tokens; ++k) {
      size_t bytes{input.GetBytes(k)};
      const char *text{input.GetText(k)};
      if (bytes == 1 && *text == '(') {
        ++nesting;
      } else if (bytes == 1 && *text == ')') {
        if (nesting == 0) {
          break;
        }
        --nesting;
      } else if (bytes == 1 && *text == ',' && nesting == 0) {
        argStart.push_back(k + 1);
      }
    }
    if (k >= tokens ||
        argStart.size() != def.argumentCount()) {
      result->push_back(text, bytes);
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
    TokenSequence repl{def.Apply(args)};
    def.set_isDisabled(true);
    TokenSequence rescanned;
    result->Append(MacroReplacement(repl, &rescanned) ? rescanned : repl);
    def.set_isDisabled(false);
  }
  return true;
}

static size_t SkipBlanks(const TokenSequence &token, size_t at) {
  for (; at < token.size(); ++at) {
    if (token.GetBytes(at) > 0 && *token.GetText(at) != ' ') {
      break;
    }
  }
  return at;
}

static std::string GetDirectiveName(const TokenSequence &line) {
  size_t tokens{line.size()};
  size_t j{SkipBlanks(line, 0)};
  if (j == tokens || line.GetString(j) != "#") {
    return ""s;
  }
  j = SkipBlanks(line, j + 1);
  if (j == tokens) {
    return ""s;
  }
  return line.GetString(j);
}

std::string Preprocessor::Directive(const TokenSequence &dir) {
  size_t tokens{dir.size()};
  size_t j{SkipBlanks(dir, 0)};
  if (j == tokens) {
    return ""s;
  }
  if (dir.GetString(j) != "#") {
    return "missing '#'"s;
  }
  j = SkipBlanks(dir, j + 1);
  if (j == tokens) {
    return ""s;
  }
  if (isdigit(*dir.GetText(j)) || *dir.GetText(j) == '"') {
    return ""s;  // TODO: treat as #line
  }
  std::string dirName{dir.GetString(j)};
  for (char &ch : dirName) {
    ch = tolower(ch);
  }
  j = SkipBlanks(dir, j + 1);
  std::string nameString;
  CharPointerWithLength nameToken;
  if (j < tokens && (isalpha(*dir.GetText(j)) || *dir.GetText(j) == '_')) {
    nameString = dir.GetString(j);
    nameToken = dir.GetToken(j);
  }
  if (dirName == "define") {
    if (nameToken.empty()) {
      return "#define: missing or invalid name"s;
    }
    // Get a pointer to a "permanent" copy of the name for use as the
    // key in the definitions_ map.
    names_.push_back(nameString);
    nameToken = CharPointerWithLength{names_.back().data(),
                                      names_.back().size()};
    definitions_.erase(nameToken);
    if (++j < tokens && dir.GetBytes(j) == 1 && *dir.GetText(j) == '(') {
      j = SkipBlanks(dir, j + 1);
      std::vector<std::string> argName;
      if (dir.GetString(j) != ")") {
        while (true) {
          std::string an{dir.GetString(j)};
          if (an.empty() || (an[0] != '_' && !isalpha(an[0]))) {
            return "#define: missing or invalid argument name"s;
          }
          argName.push_back(an);
          j = SkipBlanks(dir, j + 1);
          if (j == tokens) {
            return "#define: malformed argument list"s;
          }
          std::string punc{dir.GetString(j)};
          if (punc == ")") {
            break;
          }
          if (punc != ",") {
            return "#define: malformed argument list"s;
          }
          j = SkipBlanks(dir, j + 1);
          if (j == tokens) {
            return "#define: malformed argument list"s;
          }
        }
        if (std::set<std::string>(argName.begin(), argName.end()).size() !=
            argName.size()) {
          return "#define: argument names are not distinct"s;
        }
      }
      j = SkipBlanks(dir, j + 1);
      definitions_.emplace(
        std::make_pair(nameToken, Definition{argName, dir, j, tokens - j}));
    } else {
      definitions_.emplace(
        std::make_pair(nameToken, Definition{dir, j, tokens - j}));
    }
    return ""s;
  }
  if (dirName == "undef") {
    if (nameToken.empty()) {
      return "#undef: missing or invalid name"s;
    }
    j = SkipBlanks(dir, j + 1);
    if (j != tokens) {
      return "#undef: excess tokens at end of directive"s;
    }
    definitions_.erase(nameToken);
    return ""s;
  }
  if (dirName == "ifdef" || dirName == "ifndef") {
    if (nameToken.empty()) {
      return "#"s + dirName + ": missing name";
    }
    j = SkipBlanks(dir, j + 1);
    if (j != tokens) {
      return "#"s + dirName + ": excess tokens at end of directive";
    }
    auto it = definitions_.find(nameToken);
    if ((it != definitions_.end()) == (dirName == "ifdef")) {
      ifStack_.push(true);  // #else / #elsif allowed
      return {};
    }
    int nesting{0};
    while (std::optional<TokenSequence>
             line{prescanner_->NextTokenizedLine()}) {
      std::string dn{GetDirectiveName(*line)};
      if (dn == "ifdef" || dn == "ifndef" || dn == "if") {
        ++nesting;
      } else if (dn == "endif") {
        if (nesting-- == 0) {
          return ""s;
        }
      } else if (dn == "else" && nesting == 0) {
        ifStack_.push(false);
        return ""s;
      } // TODO: #elsif
    }
    return "#"s + dirName + ": missing #endif";
  }
  if (dirName == "else") {
    j = SkipBlanks(dir, j);
    if (j != tokens) {
      return "#else: excess tokens at end of directive"s;
    }
    if (ifStack_.empty()) {
      return "#else: no #if, #ifdef, or #ifndef"s;
    }
    if (!ifStack_.top()) {
      return "#else: already appeared in this #if, #ifdef, or #ifndef"s;
    }
    ifStack_.pop();
    int nesting{0};
    while (std::optional<TokenSequence>
             line{prescanner_->NextTokenizedLine()}) {
      std::string dn{GetDirectiveName(*line)};
      if (dn == "ifdef" || dn == "ifndef" || dn == "if") {
        ++nesting;
      } else if (dn == "endif") {
        if (nesting-- == 0) {
          return ""s;
        }
      }
    }
    return "#else: missing #endif"s;
  }
  // TODO: #if, #elsif with macro replacement on expressions
  if (dirName == "endif") {
    j = SkipBlanks(dir, j);
    if (j != tokens) {
      return "#endif: excess tokens at end of directive"s;
    }
    if (ifStack_.empty()) {
      return "#endif: no #if, #ifdef, or #ifndef"s;
    }
    ifStack_.pop();
    return ""s;
  }
  return "#"s + dirName + ": unknown or unimplemented directive";
}
}  // namespace Fortran
