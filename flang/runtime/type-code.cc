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

#include "type-code.h"

namespace Fortran::runtime {

TypeCode::TypeCode(TypeCategory f, int kind) {
  switch (f) {
  case TypeCategory::Integer:
    switch (kind) {
    case 1: raw_ = CFI_type_int8_t; break;
    case 2: raw_ = CFI_type_int16_t; break;
    case 4: raw_ = CFI_type_int32_t; break;
    case 8: raw_ = CFI_type_int64_t; break;
    case 16: raw_ = CFI_type_int128_t; break;
    }
    break;
  case TypeCategory::Real:
    switch (kind) {
    case 4: raw_ = CFI_type_float; break;
    case 8: raw_ = CFI_type_double; break;
    case 10:
    case 16: raw_ = CFI_type_long_double; break;
    }
    break;
  case TypeCategory::Complex:
    switch (kind) {
    case 4: raw_ = CFI_type_float_Complex; break;
    case 8: raw_ = CFI_type_double_Complex; break;
    case 10:
    case 16: raw_ = CFI_type_long_double_Complex; break;
    }
    break;
  case TypeCategory::Character:
    if (kind == 1) {
      raw_ = CFI_type_cptr;
    }
    break;
  case TypeCategory::Logical:
    switch (kind) {
    case 1: raw_ = CFI_type_Bool; break;
    case 2: raw_ = CFI_type_int16_t; break;
    case 4: raw_ = CFI_type_int32_t; break;
    case 8: raw_ = CFI_type_int64_t; break;
    }
    break;
  case TypeCategory::Derived: raw_ = CFI_type_struct; break;
  }
}
}
