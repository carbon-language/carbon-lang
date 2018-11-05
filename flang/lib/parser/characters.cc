// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "characters.h"
#include <cstddef>
#include <optional>

namespace Fortran::parser {

std::optional<int> UTF8CharacterBytes(const char *p) {
  if ((*p & 0x80) == 0) {
    return {1};
  }
  if ((*p & 0xf8) == 0xf0) {
    if (*p != 0xf0 && (p[1] & 0xc0) == 0x80 && (p[2] & 0xc0) == 0x80 &&
        (p[3] & 0xc0) == 0x80) {
      return {4};
    }
  } else if ((*p & 0xf0) == 0xe0) {
    if (*p != 0xe0 && (p[1] & 0xc0) == 0x80 && (p[2] & 0xc0) == 0x80) {
      return {3};
    }
  } else if ((*p & 0xe0) == 0xc0) {
    if (*p != 0xc0 && (p[1] & 0xc0) == 0x80) {
      return {2};
    }
  }
  return std::nullopt;
}

std::optional<int> EUC_JPCharacterBytes(const char *p) {
  int b1 = *p & 0xff;
  if (b1 <= 0x7f) {
    return {1};
  }
  if (b1 >= 0xa1 && b1 <= 0xfe) {
    int b2 = p[1] & 0xff;
    if (b2 >= 0xa1 && b2 <= 0xfe) {
      // JIS X 0208 (code set 1)
      return {2};
    }
  } else if (b1 == 0x8e) {
    int b2 = p[1] & 0xff;
    if (b2 >= 0xa1 && b2 <= 0xdf) {
      // upper half JIS 0201 (half-width kana, code set 2)
      return {2};
    }
  } else if (b1 == 0x8f) {
    int b2 = p[1] & 0xff;
    int b3 = p[2] & 0xff;
    if (b2 >= 0xa1 && b2 <= 0xfe && b3 >= 0xa1 && b3 <= 0xfe) {
      // JIS X 0212 (code set 3)
      return {3};
    }
  }
  return std::nullopt;
}

std::optional<std::size_t> CountCharacters(
    const char *p, std::size_t bytes, std::optional<int> (*cbf)(const char *)) {
  std::size_t chars{0};
  const char *limit{p + bytes};
  while (p < limit) {
    ++chars;
    std::optional<int> cb{cbf(p)};
    if (!cb.has_value()) {
      return std::nullopt;
    }
    p += *cb;
  }
  return {chars};
}

template<typename STRING>
std::string QuoteCharacterLiteralHelper(
    const STRING &str, bool doubleDoubleQuotes, bool doubleBackslash) {
  std::string result{'"'};
  const auto emit{[&](char ch) { result += ch; }};
  for (auto ch : str) {
    EmitQuotedChar(ch, emit, emit, doubleDoubleQuotes, doubleBackslash);
  }
  result += '"';
  return result;
}

std::string QuoteCharacterLiteral(
    const std::string &str, bool doubleDoubleQuotes, bool doubleBackslash) {
  return QuoteCharacterLiteralHelper(str, doubleDoubleQuotes, doubleBackslash);
}

std::string QuoteCharacterLiteral(
    const std::u16string &str, bool doubleDoubleQuotes, bool doubleBackslash) {
  return QuoteCharacterLiteralHelper(str, doubleDoubleQuotes, doubleBackslash);
}

std::string QuoteCharacterLiteral(
    const std::u32string &str, bool doubleDoubleQuotes, bool doubleBackslash) {
  return QuoteCharacterLiteralHelper(str, doubleDoubleQuotes, doubleBackslash);
}

std::optional<std::u32string> DecodeUTF8(const std::string &s) {
  std::u32string result;
  const std::uint8_t *p{reinterpret_cast<const std::uint8_t *>(s.data())};
  for (auto bytes{s.size()}; bytes != 0;) {
    decltype(bytes) charBytes{1};
    char32_t ch{*p++};
    if ((ch & 0xc0) > 0x40) {
      if ((ch & 0xf8) == 0xf0 && bytes >= 4 && ch > 0xf0 &&
          ((p[0] | p[1] | p[2]) & 0xc0) == 0x80) {
        charBytes = 4;
        ch = ((ch & 7) << 6) | (*p++ & 0x3f);
        ch = (ch << 6) | (*p++ & 0x3f);
        ch = (ch << 6) | (*p++ & 0x3f);
      } else if ((ch & 0xf0) == 0xe0 && bytes >= 3 && ch > 0xe0 &&
          ((p[0] | p[1]) & 0xc0) == 0x80) {
        charBytes = 3;
        ch = ((ch & 0xf) << 6) | (*p++ & 0x3f);
        ch = (ch << 6) | (*p++ & 0x3f);
      } else if ((ch & 0xe0) == 0xc0 && bytes >= 2 && ch > 0xc0 &&
          (*p & 0xc0) == 0x80) {
        charBytes = 2;
        ch = ((ch & 0x1f) << 6) | (*p++ & 0x3f);
      } else {
        return std::nullopt;  // not valid UTF-8
      }
    }
    bytes -= charBytes;
  }
  return {result};
}
}
