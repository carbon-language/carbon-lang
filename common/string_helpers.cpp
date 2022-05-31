// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/string_helpers.h"

#include <algorithm>
#include <optional>

#include "common/check.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

static constexpr llvm::StringRef TripleQuotes = R"(""")";
static constexpr llvm::StringRef HorizontalWhitespaceChars = " \t";

// Carbon only takes uppercase hex input.
static auto FromHex(char c) -> std::optional<char> {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'A' && c <= 'F') {
    return 10 + c - 'A';
  }
  return std::nullopt;
}

auto UnescapeStringLiteral(llvm::StringRef source, std::size_t hashtag_num,
                           bool is_block_string) -> std::optional<std::string> {
  std::string ret;
  std::string escape = "\\" + std::string(hashtag_num, '#');
  ret.reserve(source.size());
  size_t i = 0;
  while (i < source.size()) {
    char c = source[i];
    if (i + hashtag_num < source.size() &&
        source.slice(i, i + hashtag_num + 1).equals(escape)) {
      i += hashtag_num + 1;
      if (i == source.size()) {
        return std::nullopt;
      }
      switch (source[i]) {
        case 'n':
          ret.push_back('\n');
          break;
        case 'r':
          ret.push_back('\r');
          break;
        case 't':
          ret.push_back('\t');
          break;
        case '0':
          if (i + 1 < source.size() && llvm::isDigit(source[i + 1])) {
            // \0[0-9] is reserved.
            return std::nullopt;
          }
          ret.push_back('\0');
          break;
        case '"':
          ret.push_back('"');
          break;
        case '\'':
          ret.push_back('\'');
          break;
        case '\\':
          ret.push_back('\\');
          break;
        case 'x': {
          i += 2;
          if (i >= source.size()) {
            return std::nullopt;
          }
          std::optional<char> c1 = FromHex(source[i - 1]);
          std::optional<char> c2 = FromHex(source[i]);
          if (c1 == std::nullopt || c2 == std::nullopt) {
            return std::nullopt;
          }
          ret.push_back(16 * *c1 + *c2);
          break;
        }
        case 'u':
          CARBON_FATAL() << "\\u is not yet supported in string literals";
        case '\n':
          if (!is_block_string) {
            return std::nullopt;
          }
          break;
        default:
          // Unsupported.
          return std::nullopt;
      }
    } else if (c == '\t') {
      // Disallow non-` ` horizontal whitespace:
      // https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/lexical_conventions/whitespace.md
      // TODO: This doesn't handle unicode whitespace.
      return std::nullopt;
    } else {
      ret.push_back(c);
    }
    ++i;
  }
  return ret;
}

auto ParseBlockStringLiteral(llvm::StringRef source, std::size_t hashtag_num)
    -> ErrorOr<std::string> {
  llvm::SmallVector<llvm::StringRef> lines;
  source.split(lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  if (lines.size() < 2) {
    return Error("Too few lines");
  }

  llvm::StringRef first = lines[0];
  if (!first.consume_front(TripleQuotes)) {
    return Error("Should start with triple quotes: " + first);
  }
  first = first.rtrim(HorizontalWhitespaceChars);
  // Remaining chars, if any, are a file type indicator.
  if (first.find_first_of("\"#") != llvm::StringRef::npos ||
      first.find_first_of(HorizontalWhitespaceChars) != llvm::StringRef::npos) {
    return Error("Invalid characters in file type indicator: " + first);
  }

  llvm::StringRef last = lines[lines.size() - 1];
  const size_t last_length = last.size();
  last = last.ltrim(HorizontalWhitespaceChars);
  const size_t indent = last_length - last.size();
  if (last != TripleQuotes) {
    return Error("Should end with triple quotes: " + last);
  }

  std::string parsed;
  for (size_t i = 1; i < lines.size() - 1; ++i) {
    llvm::StringRef line = lines[i];
    const size_t first_non_ws =
        line.find_first_not_of(HorizontalWhitespaceChars);
    if (first_non_ws == llvm::StringRef::npos) {
      // Empty or whitespace-only line.
      line = "";
    } else {
      if (first_non_ws < indent) {
        return Error("Wrong indent for line: " + line + ", expected " +
                     llvm::Twine(indent));
      }
      line = line.drop_front(indent).rtrim(HorizontalWhitespaceChars);
    }
    // Unescaping with \n appended to handle things like \\<newline>.
    llvm::SmallVector<char> buffer;
    std::optional<std::string> unescaped =
        UnescapeStringLiteral((line + "\n").toStringRef(buffer), hashtag_num,
                              /*is_block_string=*/true);
    if (!unescaped.has_value()) {
      return Error("Invalid escaping in " + line);
    }
    // A \<newline> string collapses into nothing.
    if (!unescaped->empty()) {
      parsed.append(*unescaped);
    }
  }
  return parsed;
}

auto StringRefContainsPointer(llvm::StringRef ref, const char* ptr) -> bool {
  auto le = std::less_equal<const char*>();
  return le(ref.begin(), ptr) && le(ptr, ref.end());
}

}  // namespace Carbon
