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

// C and C++ API for binary-to/from-decimal conversion package.

#ifndef FORTRAN_DECIMAL_H_
#define FORTRAN_DECIMAL_H_

#include "binary-floating-point.h"
#include <stddef.h>

struct ConversionToDecimalResult {
  const char *str; /* may not be original buffer pointer; null if overflow */
  size_t length; /* not including NUL terminator */
  int decimalExponent; /* assuming decimal point to the left of first digit */
};

enum FortranRounding {
  RoundNearest, /* RN */
  RoundUp, /* RU */
  RoundDown, /* RD */
  RoundToZero, /* RZ - no rounding */
  RoundCompatible, /* RC: like RN, but ties go away from 0 */
  RoundDefault, /* RP: maps to one of the above */
};

/* The "minimize" flag causes the fewest number of output digits
 * to be emitted such that reading them back into the same binary
 * floating-point format with RoundNearest will return the same
 * value.
 */
enum DecimalConversionFlags {
  Minimize = 1, /* Minimize # of digits */
  AlwaysSign = 2, /* emit leading '+' if not negative */
};

enum BinaryConversionResultFlags {
  Exact = 0,
  Overflow = 1,
  Inexact = 2,
  Invalid = 4,
};

#ifdef __cplusplus
namespace Fortran::decimal {

template<int PREC>
ConversionToDecimalResult ConvertToDecimal(char *, size_t, int flags,
    int digits, enum FortranRounding rounding,
    BinaryFloatingPointNumber<PREC> x);

extern template ConversionToDecimalResult ConvertToDecimal<8>(char *, size_t,
    int, int, enum FortranRounding, BinaryFloatingPointNumber<8>);
extern template ConversionToDecimalResult ConvertToDecimal<11>(char *, size_t,
    int, int, enum FortranRounding, BinaryFloatingPointNumber<11>);
extern template ConversionToDecimalResult ConvertToDecimal<24>(char *, size_t,
    int, int, enum FortranRounding, BinaryFloatingPointNumber<24>);
extern template ConversionToDecimalResult ConvertToDecimal<53>(char *, size_t,
    int, int, enum FortranRounding, BinaryFloatingPointNumber<53>);
extern template ConversionToDecimalResult ConvertToDecimal<64>(char *, size_t,
    int, int, enum FortranRounding, BinaryFloatingPointNumber<64>);
extern template ConversionToDecimalResult ConvertToDecimal<112>(char *, size_t,
    int, int, enum FortranRounding, BinaryFloatingPointNumber<112>);

template<int PREC> struct ConversionToBinaryResult {
  BinaryFloatingPointNumber<PREC> binary;
  enum BinaryConversionResultFlags flags { Exact };
};

template<int PREC>
ConversionToBinaryResult<PREC> ConvertToBinary(
    const char *&, enum FortranRounding = RoundNearest);

extern template ConversionToBinaryResult<8> ConvertToBinary<8>(
    const char *&, enum FortranRounding = RoundNearest);
extern template ConversionToBinaryResult<11> ConvertToBinary<11>(
    const char *&, enum FortranRounding = RoundNearest);
extern template ConversionToBinaryResult<24> ConvertToBinary<24>(
    const char *&, enum FortranRounding = RoundNearest);
extern template ConversionToBinaryResult<53> ConvertToBinary<53>(
    const char *&, enum FortranRounding = RoundNearest);
extern template ConversionToBinaryResult<64> ConvertToBinary<64>(
    const char *&, enum FortranRounding = RoundNearest);
extern template ConversionToBinaryResult<112> ConvertToBinary<112>(
    const char *&, enum FortranRounding = RoundNearest);
}  // namespace
extern "C" {
#endif /* C++ */

ConversionToDecimalResult ConvertFloatToDecimal(
    char *, size_t, int flags, int digits, enum FortranRounding, float);
ConversionToDecimalResult ConvertDoubleToDecimal(
    char *, size_t, int flags, int digits, enum FortranRounding, double);
#if __x86_64__
ConversionToDecimalResult ConvertLongDoubleToDecimal(
    char *, size_t, int flags, int digits, enum FortranRounding, long double);
#endif

#ifdef __cplusplus
}
#endif /* C++ */
#endif /* DECIMAL_H_ */
