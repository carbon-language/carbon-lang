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

#ifndef FORTRAN_COMMON_FORTRAN_H_
#define FORTRAN_COMMON_FORTRAN_H_

// Fortran language concepts that are used in many phases are defined
// once here to avoid redundancy and needless translation.

#include "idioms.h"

namespace Fortran::common {

// Fortran has five kinds of intrinsic data, and the derived types.
ENUM_CLASS(TypeCategory, Integer, Real, Complex, Character, Logical, Derived)

// Kinds of IMPORT statements. Default means IMPORT or IMPORT :: names.
ENUM_CLASS(ImportKind, Default, Only, None, All)

// Type parameters can be kind or len.
ENUM_CLASS(TypeParamKindOrLen, Kind, Len)

}  // namespace Fortran::common
#endif  // FORTRAN_COMMON_FORTRAN_H_
