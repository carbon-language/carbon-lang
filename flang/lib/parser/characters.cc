// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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
#include "../common/idioms.h"
#include <cstddef>
#include <optional>
#include <type_traits>

namespace Fortran::parser {

std::optional<int> UTF_8CharacterBytes(const char *p) {
  if ((*p & 0x80) == 0) {
    return 1;
  }
  if ((*p & 0xf8) == 0xf0) {
    if ((*p & 0x07) != 0 && (p[1] & 0xc0) == 0x80 && (p[2] & 0xc0) == 0x80 &&
        (p[3] & 0xc0) == 0x80) {
      return 4;
    }
  } else if ((*p & 0xf0) == 0xe0) {
    if ((*p & 0x0f) != 0 && (p[1] & 0xc0) == 0x80 && (p[2] & 0xc0) == 0x80) {
      return 3;
    }
  } else if ((*p & 0xe0) == 0xc0) {
    if ((*p & 0x1f) != 0 && (p[1] & 0xc0) == 0x80) {
      return 2;
    }
  }
  return std::nullopt;
}

std::optional<int> EUC_JPCharacterBytes(const char *p) {
  int b1 = *p & 0xff;
  if (b1 <= 0x7f) {
    return 1;
  }
  if (b1 >= 0xa0 && b1 <= 0xfe) {
    int b2 = p[1] & 0xff;
    if (b2 >= 0xa0 && b2 <= 0xfe) {
      // JIS X 0208 (code set 1)
      return 2;
    }
  } else if (b1 == 0x8e) {
    int b2 = p[1] & 0xff;
    if (b2 >= 0xa0 && b2 <= 0xdf) {
      // upper half JIS 0201 (half-width kana, code set 2)
      return 2;
    }
  } else if (b1 == 0x8f) {
    int b2 = p[1] & 0xff;
    int b3 = p[2] & 0xff;
    if (b2 >= 0xa0 && b2 <= 0xfe && b3 >= 0xa0 && b3 <= 0xfe) {
      // JIS X 0212 (code set 3)
      return 3;
    }
  }
  return std::nullopt;
}

static std::optional<int> One(const char *) { return 1; }

static std::optional<int> (*CharacterCounter(Encoding encoding))(const char *) {
  switch (encoding) {
  case Encoding::UTF_8: return UTF_8CharacterBytes;
  case Encoding::EUC_JP: return EUC_JPCharacterBytes;
  default: return One;
  }
}

std::optional<int> CountCharacters(
    const char *p, std::size_t bytes, Encoding encoding) {
  std::size_t chars{0};
  const char *limit{p + bytes};
  std::optional<int> (*cbf)(const char *){CharacterCounter(encoding)};
  while (p < limit) {
    if (std::optional<int> cb{cbf(p)}) {
      p += *cb;
      ++chars;
    } else {
      return std::nullopt;
    }
  }
  if (p == limit) {
    return chars;
  } else {
    return std::nullopt;
  }
}

template<typename STRING>
std::string QuoteCharacterLiteralHelper(
    const STRING &str, bool backslashEscapes, Encoding encoding) {
  std::string result{'"'};
  const auto emit{[&](char ch) { result += ch; }};
  for (auto ch : str) {
    using CharT = std::decay_t<decltype(ch)>;
    char32_t ch32{static_cast<std::make_unsigned_t<CharT>>(ch)};
    if (ch32 == static_cast<unsigned char>('"')) {
      emit('"');  // double the " when it appears in the text
    }
    EmitQuotedChar(ch32, emit, emit, backslashEscapes, encoding);
  }
  result += '"';
  return result;
}

std::string QuoteCharacterLiteral(
    const std::string &str, bool backslashEscapes, Encoding encoding) {
  return QuoteCharacterLiteralHelper(str, backslashEscapes, encoding);
}

std::string QuoteCharacterLiteral(
    const std::u16string &str, bool backslashEscapes, Encoding encoding) {
  return QuoteCharacterLiteralHelper(str, backslashEscapes, encoding);
}

std::string QuoteCharacterLiteral(
    const std::u32string &str, bool backslashEscapes, Encoding encoding) {
  return QuoteCharacterLiteralHelper(str, backslashEscapes, encoding);
}

EncodedCharacter EncodeLATIN_1(char32_t codepoint) {
  CHECK(codepoint <= 0xff);
  EncodedCharacter result;
  result.buffer[0] = codepoint;
  result.bytes = 1;
  return result;
}

EncodedCharacter EncodeUTF_8(char32_t codepoint) {
  // N.B. char32_t is unsigned
  EncodedCharacter result;
  if (codepoint <= 0x7f) {
    result.buffer[0] = codepoint;
    result.bytes = 1;
  } else if (codepoint <= 0x7ff) {
    result.buffer[0] = 0xc0 | (codepoint >> 6);
    result.buffer[1] = 0x80 | (codepoint & 0x3f);
    result.bytes = 2;
  } else if (codepoint <= 0xffff) {
    result.buffer[0] = 0xe0 | (codepoint >> 12);
    result.buffer[1] = 0x80 | ((codepoint >> 6) & 0x3f);
    result.buffer[2] = 0x80 | (codepoint & 0x3f);
    result.bytes = 3;
  } else {
    // UCS actually only goes up to 0x10ffff but the
    // UTF-8 encoding handles 21 bits.
    CHECK(codepoint <= 0x1fffff);
    result.buffer[0] = 0xf0 | (codepoint >> 18);
    result.buffer[1] = 0x80 | ((codepoint >> 12) & 0x3f);
    result.buffer[2] = 0x80 | ((codepoint >> 6) & 0x3f);
    result.buffer[3] = 0x80 | (codepoint & 0x3f);
    result.bytes = 4;
  }
  return result;
}

EncodedCharacter EncodeEUC_JP(char32_t codepoint) {
  // Assume JIS X 0208 (TODO: others)
  CHECK(codepoint <= 0x6e6e);
  EncodedCharacter result;
  if (codepoint <= 0x7f) {
    result.buffer[0] = codepoint;
    result.bytes = 1;
  } else {
    result.buffer[0] = 0x80 | (codepoint >> 8);
    result.buffer[1] = 0x80 | (codepoint & 0x7f);
    result.bytes = 2;
  }
  return result;
}

EncodedCharacter EncodeCharacter(Encoding encoding, char32_t codepoint) {
  switch (encoding) {
  case Encoding::LATIN_1: return EncodeLATIN_1(codepoint);
  case Encoding::UTF_8: return EncodeUTF_8(codepoint);
  case Encoding::EUC_JP: return EncodeEUC_JP(codepoint);
  default: CRASH_NO_CASE;
  }
}

DecodedCharacter DecodeUTF_8Character(const char *cp, std::size_t bytes) {
  auto p{reinterpret_cast<const std::uint8_t *>(cp)};
  char32_t ch{*p};
  if (ch <= 0x7f) {
    return {ch, 1};
  } else if ((ch & 0xf8) == 0xf0 && bytes >= 4 && ch > 0xf0 &&
      ((p[1] | p[2] | p[3]) & 0xc0) == 0x80) {
    ch = ((ch & 7) << 6) | (p[1] & 0x3f);
    ch = (ch << 6) | (p[2] & 0x3f);
    ch = (ch << 6) | (p[3] & 0x3f);
    return {ch, 4};
  } else if ((ch & 0xf0) == 0xe0 && bytes >= 3 && ch > 0xe0 &&
      ((p[1] | p[2]) & 0xc0) == 0x80) {
    ch = ((ch & 0xf) << 6) | (p[1] & 0x3f);
    ch = (ch << 6) | (p[2] & 0x3f);
    return {ch, 3};
  } else if ((ch & 0xe0) == 0xc0 && bytes >= 2 && ch > 0xc0 &&
      (p[1] & 0xc0) == 0x80) {
    ch = ((ch & 0x1f) << 6) | (p[1] & 0x3f);
    return {ch, 2};
  } else {
    return {};  // not valid UTF-8
  }
}

DecodedCharacter DecodeEUC_JPCharacter(const char *cp, std::size_t bytes) {
  auto p{reinterpret_cast<const std::uint8_t *>(cp)};
  char32_t ch{*p};
  if (ch <= 0x7f) {
    return {ch, 1};
  } else if (ch >= 0xa0 && ch <= 0xfe && bytes >= 2 && p[1] >= 0xa0 &&
      p[1] <= 0xfe) {
    ch = ((ch << 8) | p[1]) & 0x7f7f;  // JIS X 0208
    return {ch, 2};
  } else if (ch == 0x8e && bytes >= 2 && p[1] >= 0xa0 && p[1] <= 0xdf) {
    return {p[1], 2};  // JIS X 0201
  } else if (ch == 0x8f && bytes >= 3 && p[1] >= 0xa0 && p[1] <= 0xfe &&
      p[2] >= 0xa0 && p[2] <= 0xfe) {
    ch = ((p[1] << 8) | p[1]) & 0x7f7f;  // JIS X 0212
    return {ch, 3};
  } else {
    return {};  // not valid EUC_JP
  }
}

DecodedCharacter DecodeLATIN1Character(const char *cp) {
  return {*reinterpret_cast<const std::uint8_t *>(cp), 1};
}

static DecodedCharacter DecodeEscapedCharacter(
    const char *cp, std::size_t bytes) {
  if (cp[0] == '\\' && bytes >= 2) {
    if (std::optional<char> escChar{BackslashEscapeValue(cp[1])}) {
      return {static_cast<unsigned char>(*escChar), 2};
    }
    if (IsOctalDigit(cp[1])) {
      std::size_t maxDigits{static_cast<std::size_t>(cp[1] > '3' ? 2 : 3)};
      std::size_t maxLen{std::max(maxDigits + 1, bytes)};
      char32_t code{static_cast<char32_t>(cp[1] - '0')};
      std::size_t len{2};  // so far
      for (; len < maxLen && IsOctalDigit(cp[len]); ++len) {
        code = 8 * code + DecimalDigitValue(cp[len]);
      }
      return {code, static_cast<int>(len)};
    } else if (bytes >= 4 && ToLowerCaseLetter(cp[1]) == 'x' &&
        IsHexadecimalDigit(cp[2]) && IsHexadecimalDigit(cp[3])) {
      return {static_cast<char32_t>(16 * HexadecimalDigitValue(cp[2]) +
                  HexadecimalDigitValue(cp[3])),
          4};
    }
  }
  return {static_cast<unsigned char>(cp[0]), 1};
}

static DecodedCharacter DecodeEscapedCharacters(
    Encoding encoding, const char *cp, std::size_t bytes) {
  char buffer[4];
  int count[4];
  std::size_t at{0}, len{0};
  for (; len < 4 && at < bytes; ++len) {
    DecodedCharacter code{DecodeEscapedCharacter(cp + at, bytes - at)};
    buffer[len] = code.codepoint;
    at += code.bytes;
    count[len] = at;
  }
  DecodedCharacter code{DecodeCharacter(encoding, buffer, len, false)};
  if (code.bytes > 0) {
    code.bytes = count[code.bytes - 1];
  } else {
    code.codepoint = static_cast<unsigned char>(buffer[0]);
    code.bytes = count[0];
  }
  return code;
}

DecodedCharacter DecodeCharacter(Encoding encoding, const char *cp,
    std::size_t bytes, bool backslashEscapes) {
  if (backslashEscapes && bytes >= 2 && *cp == '\\') {
    return DecodeEscapedCharacters(encoding, cp, bytes);
  } else {
    switch (encoding) {
    case Encoding::LATIN_1:
      if (bytes >= 1) {
        return DecodeLATIN1Character(cp);
      } else {
        return {};
      }
    case Encoding::UTF_8: return DecodeUTF_8Character(cp, bytes);
    case Encoding::EUC_JP: return DecodeEUC_JPCharacter(cp, bytes);
    default: CRASH_NO_CASE;
    }
  }
}

std::u32string DecodeUTF_8(const std::string &s) {
  std::u32string result;
  const char *p{s.c_str()};
  for (auto bytes{s.size()}; bytes != 0;) {
    DecodedCharacter decoded{DecodeUTF_8Character(p, bytes)};
    if (decoded.bytes > 0) {
      if (static_cast<std::size_t>(decoded.bytes) <= bytes) {
        result.append(1, decoded.codepoint);
        bytes -= decoded.bytes;
        p += decoded.bytes;
        continue;
      }
    }
    result.append(1, static_cast<uint8_t>(*p));
    ++p;
    --bytes;
  }
  return result;
}

std::u16string DecodeEUC_JP(const std::string &s) {
  std::u16string result;
  const char *p{s.c_str()};
  for (auto bytes{s.size()}; bytes != 0;) {
    DecodedCharacter decoded{DecodeEUC_JPCharacter(p, bytes)};
    if (decoded.bytes > 0) {
      if (static_cast<std::size_t>(decoded.bytes) <= bytes) {
        result.append(1, decoded.codepoint);
        bytes -= decoded.bytes;
        p += decoded.bytes;
        continue;
      }
    }
    result.append(1, static_cast<uint8_t>(*p));
    ++p;
    --bytes;
  }
  return result;
}

}
