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

// Provides implementations of intrinsic functions operating on character
// scalars. No assumption is made regarding character encodings other than they
// must be compatible with ASCII (else, NEW_LINE, ACHAR and IACHAR need to be
// adapted).

namespace Fortran::evaluate {

template<int KIND> class CharacterUtils {
  using Character = Scalar<Type<TypeCategory::Character, KIND>>;
  using CharT = typename Character::value_type;

public:
  // CHAR also implements ACHAR under assumption that character encodings
  // contain ASCII
  static Character CHAR(std::uint64_t code) {
    return Character{{static_cast<CharT>(code)}};
  }

  // ICHAR also implements IACHAR under assumption that character encodings
  // contain ASCII
  static std::int64_t ICHAR(const Character &c) {
    CHECK(c.length() == 1);
    if constexpr (std::is_same_v<CharT, char>) {
      // char may be signed, so cast it first to unsigned to avoid having
      // ichar(char(128_4)) returning -128
      return static_cast<unsigned char>(c[0]);
    } else {
      return c[0];
    }
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

  // Resize adds spaces on the right if the new size is bigger than the
  // original, or by trimming the rightmost characters otherwise.
  static Character Resize(const Character &str, std::size_t newLength) {
    auto oldLength{str.length()};
    if (newLength > oldLength) {
      return str + Character(newLength - oldLength, Space());
    } else {
      return str.substr(0, newLength);
    }
  }

private:
  // Following helpers assume that character encodings contain ASCII
  static constexpr CharT Space() { return 0x20; }
  static constexpr CharT NewLine() { return 0x0a; }
};

}

#endif  // FORTRAN_EVALUATE_CHARACTER_H_
