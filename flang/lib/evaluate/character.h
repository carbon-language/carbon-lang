// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_EVALUATE_CHARACTER_H_
#define FORTRAN_EVALUATE_CHARACTER_H_

#include "type.h"
#include <string>

// character(1) is ISO/IEC 646:1991 (ASCII)
//
// character(2) is a variant of EUC-JP: The code points are the same as EUC-JP,
// but the encoding has a two bytes fix-size (EUC-JP encoding has a variable
// length). The one byte EUC-JP character representations are simply
// zero-extended to two byte representations. The three bytes character
// representation of EUC-JP (JIS X 0212) are not supported in this internal
// encoding.
//
// character(4) is ISO/IEC 10646 UCS-4 (~ UTF-32)

namespace Fortran::evaluate {

template<int KIND> class CharacterUtils {
  using Character = Scalar<Type<TypeCategory::Character, KIND>>;
  using CharT = typename Character::value_type;

public:
  static constexpr bool IsValidCharacterCode(std::uint64_t code) {
    if constexpr (KIND == 1) {
      return IsValidASCII(code);
    } else if constexpr (KIND == 2) {
      return IsValidInternalEUC_JP(code);
    } else if constexpr (KIND == 4) {
      return IsValidUCS4(code);
    } else {
      static_assert(true, "bad character kind");
    }
  }

  // CHAR also implements ACHAR
  static Character CHAR(std::uint64_t code) {
    return Character{{static_cast<CharT>(code)}};
  }

  // ICHAR also implements IACHAR
  static int ICHAR(const Character &c) {
    CHECK(c.length() == 1);
    return c[0];
  }

  static Character NEW_LINE() { return Character{{NewLine()}}; }

  static Character ADJUSTL(const Character &str) {
    auto pos{str.find_first_not_of(Space())};
    if (pos != Character::npos && pos != 0) {
      return Character{str.substr(pos) + Character(pos, Space())};
    }
    // else empty or only spaces, or no leading spaces
    return str;
  }

  static Character ADJUSTR(const Character &str) {
    auto pos{str.find_last_not_of(Space())};
    if (pos != Character::npos && pos != str.length() - 1) {
      auto delta{str.length() - 1 - pos};
      return Character{Character(delta, Space()) + str.substr(0, pos + 1)};
    }
    // else empty or only spaces, or no trailing spaces
    return str;
  }

private:
  static constexpr bool IsValidASCII(std::uint64_t code) { return code < 128; }
  static constexpr bool IsValidInternalEUC_JP(std::uint64_t code) {
    std::uint16_t hi{static_cast<std::uint16_t>(code >> 8)};
    std::uint16_t lo{static_cast<std::uint16_t>(code & 0xff)};
    return IsValidASCII(code) ||
        (code < 0xffff &&
            ((0xa1 <= hi && hi <= 0Xfe && 0xa1 <= lo && lo <= 0Xfe) ||
                (hi == 0X8e && 0xa1 <= lo && lo <= 0Xdf)));
  }
  static constexpr bool IsValidUCS4(std::uint64_t code) {
    return code < 0xd800 || (0xdc00 < code && code <= 0x10ffff);
  }
  static constexpr CharT Space() { return 0x20; }
  static constexpr CharT NewLine() { return 0x0a; }
};

}

#endif  // FORTRAN_EVALUATE_CHARACTER_H_
