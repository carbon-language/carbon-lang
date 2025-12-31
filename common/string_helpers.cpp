// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/string_helpers.h"

#include <string.h>
#include <sys/types.h>

#include <algorithm>
#include <array>
#include <optional>
#include <string>

#include "common/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"

namespace Carbon {

static constexpr llvm::StringRef TripleQuotes = "'''";
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

auto UnescapeStringLiteral(llvm::StringRef source, const int hashtag_num,
                           bool is_block_string) -> std::optional<std::string> {
  std::string ret;
  ret.reserve(source.size());
  std::string escape = "\\";
  escape.resize(hashtag_num + 1, '#');
  size_t i = 0;
  while (i < source.size()) {
    char c = source[i];
    if (i + hashtag_num < source.size() &&
        source.slice(i, i + hashtag_num + 1) == escape) {
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
        case 'u': {
          ++i;
          if (i >= source.size() || source[i] != '{') {
            return std::nullopt;
          }
          unsigned int unicode_int = 0;
          ++i;
          int original_i = i;
          while (i < source.size() && source[i] != '}') {
            std::optional<char> hex_val = FromHex(source[i]);
            if (hex_val == std::nullopt) {
              return std::nullopt;
            }
            unicode_int = unicode_int << 4;
            unicode_int += hex_val.value();
            ++i;
            if (i - original_i > 8) {
              return std::nullopt;
            }
          }
          if (i >= source.size()) {
            return std::nullopt;
          }
          if (i - original_i == 0) {
            return std::nullopt;
          }
          char utf8_buf[4];
          char* utf8_end = &utf8_buf[0];
          if (!llvm::ConvertCodePointToUTF8(unicode_int, utf8_end)) {
            return std::nullopt;
          }
          ret.append(utf8_buf, utf8_end - utf8_buf);
          break;
        }
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

auto ParseBlockStringLiteral(llvm::StringRef source, const int hashtag_num)
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
  auto le = std::less_equal<>();
  return le(ref.begin(), ptr) && le(ptr, ref.end());
}

auto BuildCStrArgs(llvm::StringRef tool_path,
                   llvm::ArrayRef<llvm::StringRef> args,
                   llvm::OwningArrayRef<char>& cstr_arg_storage)
    -> llvm::SmallVector<const char*, 64> {
  return BuildCStrArgs(tool_path, /*prefix_args=*/{}, args, cstr_arg_storage);
}

auto BuildCStrArgs(llvm::StringRef tool_path,
                   llvm::ArrayRef<std::string> prefix_args,
                   llvm::ArrayRef<llvm::StringRef> args,
                   llvm::OwningArrayRef<char>& cstr_arg_storage)
    -> llvm::SmallVector<const char*, 64> {
  // Render the arguments into null-terminated C-strings. Command lines can get
  // quite long in build systems so this tries to minimize the memory allocation
  // overhead.

  // Precompute the total C-string data size needed.
  int total_size = tool_path.size() + 1;
  for (llvm::StringRef arg : args) {
    // Accumulate both the string size and a null terminator byte.
    total_size += arg.size() + 1;
  }

  // Allocate one chunk of storage for the actual C-strings, and reserve a
  // vector of pointers into the storage.
  cstr_arg_storage = llvm::OwningArrayRef<char>(total_size);
  ssize_t i = 0;
  auto make_cstr = [&](llvm::StringRef arg) {
    char* cstr = &cstr_arg_storage[i];
    memcpy(cstr, arg.data(), arg.size());
    cstr[arg.size()] = '\0';
    i += arg.size() + 1;
    return cstr;
  };

  llvm::SmallVector<const char*, 64> cstr_args;
  cstr_args.reserve(1 + prefix_args.size() + args.size());
  cstr_args.push_back(make_cstr(tool_path));
  for (const std::string& prefix_arg : prefix_args) {
    cstr_args.push_back(prefix_arg.c_str());
  }
  for (llvm::StringRef arg : args) {
    cstr_args.push_back(make_cstr(arg));
  }

  return cstr_args;
}

}  // namespace Carbon
