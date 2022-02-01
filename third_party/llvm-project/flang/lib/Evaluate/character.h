//===-- lib/Evaluate/character.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_CHARACTER_H_
#define FORTRAN_EVALUATE_CHARACTER_H_

#include "flang/Evaluate/type.h"
#include <string>

// Provides implementations of intrinsic functions operating on character
// scalars. No assumption is made regarding character encodings other than they
// must be compatible with ASCII (else, NEW_LINE, ACHAR and IACHAR need to be
// adapted).

namespace Fortran::evaluate {

template <int KIND> class CharacterUtils {
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

  static ConstantSubscript INDEX(
      const Character &str, const Character &substr, bool back = false) {
    auto pos{back ? str.rfind(substr) : str.find(substr)};
    return static_cast<ConstantSubscript>(pos == str.npos ? 0 : pos + 1);
  }

  static ConstantSubscript SCAN(
      const Character &str, const Character &set, bool back = false) {
    auto pos{back ? str.find_last_of(set) : str.find_first_of(set)};
    return static_cast<ConstantSubscript>(pos == str.npos ? 0 : pos + 1);
  }

  static ConstantSubscript VERIFY(
      const Character &str, const Character &set, bool back = false) {
    auto pos{back ? str.find_last_not_of(set) : str.find_first_not_of(set)};
    return static_cast<ConstantSubscript>(pos == str.npos ? 0 : pos + 1);
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

  static ConstantSubscript LEN_TRIM(const Character &str) {
    return VERIFY(str, Character{' '}, true);
  }

  static Character REPEAT(const Character &str, ConstantSubscript ncopies) {
    Character result;
    if (!str.empty()) {
      while (ncopies-- > 0) {
        result += str;
      }
    }
    return result;
  }

  static Character TRIM(const Character &str) {
    return str.substr(0, LEN_TRIM(str));
  }

private:
  // Following helpers assume that character encodings contain ASCII
  static constexpr CharT Space() { return 0x20; }
  static constexpr CharT NewLine() { return 0x0a; }
};

} // namespace Fortran::evaluate

#endif // FORTRAN_EVALUATE_CHARACTER_H_
