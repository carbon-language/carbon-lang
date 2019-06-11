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
  if (b1 >= 0xa1 && b1 <= 0xfe) {
    int b2 = p[1] & 0xff;
    if (b2 >= 0xa1 && b2 <= 0xfe) {
      // JIS X 0208 (code set 1)
      return 2;
    }
  } else if (b1 == 0x8e) {
    int b2 = p[1] & 0xff;
    if (b2 >= 0xa1 && b2 <= 0xdf) {
      // upper half JIS 0201 (half-width kana, code set 2)
      return 2;
    }
  } else if (b1 == 0x8f) {
    int b2 = p[1] & 0xff;
    int b3 = p[2] & 0xff;
    if (b2 >= 0xa1 && b2 <= 0xfe && b3 >= 0xa1 && b3 <= 0xfe) {
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

std::optional<int> CharacterBytes(const char *p, Encoding encoding) {
  return CharacterCounter(encoding)(p);
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
std::string QuoteCharacterLiteralHelper(const STRING &str,
    bool doubleDoubleQuotes, bool doubleBackslash, Encoding encoding) {
  std::string result{'"'};
  const auto emit{[&](char ch) { result += ch; }};
  for (auto ch : str) {
    using CharT = std::decay_t<decltype(ch)>;
    if constexpr (std::is_same_v<char, CharT>) {
      // char may be signed depending on host.
      char32_t ch32{static_cast<unsigned char>(ch)};
      EmitQuotedChar(
          ch32, emit, emit, doubleDoubleQuotes, doubleBackslash, encoding);
    } else {
      char32_t ch32{ch};
      EmitQuotedChar(
          ch32, emit, emit, doubleDoubleQuotes, doubleBackslash, encoding);
    }
  }
  result += '"';
  return result;
}

std::string QuoteCharacterLiteral(const std::string &str,
    bool doubleDoubleQuotes, bool doubleBackslash, Encoding encoding) {
  return QuoteCharacterLiteralHelper(
      str, doubleDoubleQuotes, doubleBackslash, encoding);
}

std::string QuoteCharacterLiteral(const std::u16string &str,
    bool doubleDoubleQuotes, bool doubleBackslash, Encoding encoding) {
  return QuoteCharacterLiteralHelper(
      str, doubleDoubleQuotes, doubleBackslash, encoding);
}

std::string QuoteCharacterLiteral(const std::u32string &str,
    bool doubleDoubleQuotes, bool doubleBackslash, Encoding encoding) {
  return QuoteCharacterLiteralHelper(
      str, doubleDoubleQuotes, doubleBackslash, encoding);
}

EncodedCharacter EncodeLATIN_1(char codepoint) {
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

EncodedCharacter EncodeEUC_JP(char16_t codepoint) {
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
  int charBytes{1};
  if (ch >= 0x80) {
    if ((ch & 0xf8) == 0xf0 && bytes >= 4 && ch > 0xf0 &&
        ((p[1] | p[2] | p[3]) & 0xc0) == 0x80) {
      charBytes = 4;
      ch = ((ch & 7) << 6) | (p[1] & 0x3f);
      ch = (ch << 6) | (p[2] & 0x3f);
      ch = (ch << 6) | (p[3] & 0x3f);
    } else if ((ch & 0xf0) == 0xe0 && bytes >= 3 && ch > 0xe0 &&
        ((p[1] | p[2]) & 0xc0) == 0x80) {
      charBytes = 3;
      ch = ((ch & 0xf) << 6) | (p[1] & 0x3f);
      ch = (ch << 6) | (p[2] & 0x3f);
    } else if ((ch & 0xe0) == 0xc0 && bytes >= 2 && ch > 0xc0 &&
        (p[1] & 0xc0) == 0x80) {
      charBytes = 2;
      ch = ((ch & 0x1f) << 6) | (p[1] & 0x3f);
    } else {
      return {};  // not valid UTF-8
    }
  }
  return {ch, charBytes};
}

DecodedCharacter DecodeEUC_JPCharacter(const char *cp, std::size_t bytes) {
  auto p{reinterpret_cast<const std::uint8_t *>(cp)};
  char32_t ch{*p};
  int charBytes{1};
  if (ch >= 0x80) {
    if (bytes >= 2 && ch == 0x8e && p[1] >= 0xa1 && p[1] <= 0xdf) {
      charBytes = 2;  // JIS X 0201
      ch = p[1];
    } else if (bytes >= 3 && ch == 0x8f && p[1] >= 0xa1 && p[1] <= 0xfe &&
        p[2] >= 0xa1 && p[2] <= 0xfe) {
      charBytes = 3;  // JIS X 0212
      ch = (p[1] & 0x7f) << 8 | (p[1] & 0x7f);
    } else if (bytes >= 2 && ch >= 0xa1 && ch <= 0xfe && p[1] >= 0x1 &&
        p[1] <= 0xfe) {
      charBytes = 2;  // JIS X 0208
      ch = ((ch & 0x7f) << 8) | (p[1] & 0x7f);
    } else {
      return {};
    }
  }
  return {ch, charBytes};
}

DecodedCharacter DecodeLATIN1Character(const char *cp) {
  return {*reinterpret_cast<const std::uint8_t *>(cp), 1};
}

DecodedCharacter DecodeCharacter(
    Encoding encoding, const char *cp, std::size_t bytes) {
  switch (encoding) {
  case Encoding::LATIN_1: return DecodeLATIN1Character(cp);
  case Encoding::UTF_8: return DecodeUTF_8Character(cp, bytes);
  case Encoding::EUC_JP: return DecodeEUC_JPCharacter(cp, bytes);
  default: CRASH_NO_CASE;
  }
}

std::u32string DecodeUTF_8(const std::string &s) {
  std::u32string result;
  const char *p{s.c_str()};
  for (auto bytes{s.size()}; bytes != 0;) {
    DecodedCharacter decoded{DecodeUTF_8Character(p, bytes)};
    if (decoded.bytes > 0) {
      if (static_cast<std::size_t>(decoded.bytes) <= bytes) {
        result.append(1, decoded.unicode);
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
        result.append(1, decoded.unicode);
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
