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

#ifndef FORTRAN_RUNTIME_TYPE_CODE_H_
#define FORTRAN_RUNTIME_TYPE_CODE_H_

#include "../include/flang/ISO_Fortran_binding.h"

namespace Fortran::runtime {

class TypeCode {
public:
  enum class Form { Integer, Real, Complex, Character, Logical, Derived };

  TypeCode() {}
  explicit TypeCode(ISO::CFI_type_t t) : raw_{t} {}
  TypeCode(Form, int);

  int raw() const { return raw_; }

  constexpr bool IsValid() const {
    return raw_ >= CFI_type_signed_char && raw_ <= CFI_type_struct;
  }
  constexpr bool IsInteger() const {
    return raw_ >= CFI_type_signed_char && raw_ <= CFI_type_ptrdiff_t;
  }
  constexpr bool IsReal() const {
    return raw_ >= CFI_type_float && raw_ <= CFI_type_long_double;
  }
  constexpr bool IsComplex() const {
    return raw_ >= CFI_type_float_Complex &&
        raw_ <= CFI_type_long_double_Complex;
  }
  constexpr bool IsCharacter() const { return raw_ == CFI_type_cptr; }
  constexpr bool IsLogical() const { return raw_ == CFI_type_Bool; }
  constexpr bool IsDerived() const { return raw_ == CFI_type_struct; }

  constexpr bool IsIntrinsic() const { return IsValid() && !IsDerived(); }

  constexpr Form GetForm() const {
    if (IsInteger()) {
      return Form::Integer;
    }
    if (IsReal()) {
      return Form::Real;
    }
    if (IsComplex()) {
      return Form::Complex;
    }
    if (IsCharacter()) {
      return Form::Character;
    }
    if (IsLogical()) {
      return Form::Logical;
    }
    return Form::Derived;
  }

private:
  ISO::CFI_type_t raw_{CFI_type_other};
};
}  // namespace Fortran::runtime
#endif  // FORTRAN_RUNTIME_TYPE_CODE_H_
