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

#ifndef FORTRAN_COMMON_FORTRAN_H_
#define FORTRAN_COMMON_FORTRAN_H_

// Fortran language concepts that are used in many phases are defined
// once here to avoid redundancy and needless translation.

#include "idioms.h"
#include <cinttypes>

namespace Fortran::common {

// Fortran has five kinds of intrinsic data types, plus the derived types.
ENUM_CLASS(TypeCategory, Integer, Real, Complex, Character, Logical, Derived)

constexpr bool IsNumericTypeCategory(TypeCategory category) {
  return category == TypeCategory::Integer || category == TypeCategory::Real ||
      category == TypeCategory::Complex;
}

// Kinds of IMPORT statements. Default means IMPORT or IMPORT :: names.
ENUM_CLASS(ImportKind, Default, Only, None, All)

// The attribute on a type parameter can be KIND or LEN.
ENUM_CLASS(TypeParamAttr, Kind, Len)

ENUM_CLASS(RelationalOperator, LT, LE, EQ, NE, GE, GT)

ENUM_CLASS(Intent, Default, In, Out, InOut)

ENUM_CLASS(IoStmtKind, None, Backspace, Close, Endfile, Flush, Inquire, Open,
    Print, Read, Rewind, Wait, Write)

// Union of specifiers for all I/O statements.
ENUM_CLASS(IoSpecKind, Access, Action, Advance, Asynchronous, Blank, Decimal,
    Delim, Direct, Encoding, End, Eor, Err, Exist, File, Fmt, Form, Formatted,
    Id, Iomsg, Iostat, Name, Named, Newunit, Nextrec, Nml, Number, Opened, Pad,
    Pending, Pos, Position, Read, Readwrite, Rec, Recl, Round, Sequential, Sign,
    Size, Status, Stream, Unformatted, Unit, Write,
    Convert,  // nonstandard
    Dispose,  // nonstandard
)
}
#endif  // FORTRAN_COMMON_FORTRAN_H_
