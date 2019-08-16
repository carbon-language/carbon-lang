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

// Binary-to-decimal conversions (formatting) produce a sequence of decimal
// digit characters in a NUL-terminated user-supplied buffer that constitute
// a decimal fraction (or zero), accompanied by a decimal exponent that
// you'll get to adjust and format yourself.  There can be a leading sign
// character.
// Negative zero is "-0".  The result can also be "NaN", "Inf", "+Inf",
// or "-Inf".
// If the conversion can't fit in the user-supplied buffer, a null pointer
// is returned.

#ifdef __cplusplus
namespace Fortran::decimal {
#endif

enum ConversionResultFlags {
  Exact = 0,
  Overflow = 1,
  Inexact = 2,
  Invalid = 4,
};

struct ConversionToDecimalResult {
  const char *str; /* may not be original buffer pointer; null if overflow */
  size_t length; /* does not include NUL terminator */
  int decimalExponent; /* assuming decimal point to the left of first digit */
  enum ConversionResultFlags flags;
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

#ifdef __cplusplus
template<int PREC>
ConversionToDecimalResult ConvertToDecimal(char *, size_t,
    DecimalConversionFlags, int digits, enum FortranRounding rounding,
    BinaryFloatingPointNumber<PREC> x);

extern template ConversionToDecimalResult ConvertToDecimal<8>(char *, size_t,
    enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<8>);
extern template ConversionToDecimalResult ConvertToDecimal<11>(char *, size_t,
    enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<11>);
extern template ConversionToDecimalResult ConvertToDecimal<24>(char *, size_t,
    enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<24>);
extern template ConversionToDecimalResult ConvertToDecimal<53>(char *, size_t,
    enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<53>);
extern template ConversionToDecimalResult ConvertToDecimal<64>(char *, size_t,
    enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<64>);
extern template ConversionToDecimalResult ConvertToDecimal<112>(char *, size_t,
    enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<112>);

template<int PREC> struct ConversionToBinaryResult {
  BinaryFloatingPointNumber<PREC> binary;
  enum ConversionResultFlags flags { Exact };
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
}  // namespace Fortran::decimal
extern "C" {
#define NS(x) Fortran::decimal::x
#else
#define NS(x) x
#endif /* C++ */

NS(ConversionToDecimalResult)
ConvertFloatToDecimal(char *, size_t, enum NS(DecimalConversionFlags),
    int digits, enum NS(FortranRounding), float);
NS(ConversionToDecimalResult)
ConvertDoubleToDecimal(char *, size_t, enum NS(DecimalConversionFlags),
    int digits, enum NS(FortranRounding), double);
#if __x86_64__
NS(ConversionToDecimalResult)
ConvertLongDoubleToDecimal(char *, size_t, enum NS(DecimalConversionFlags),
    int digits, enum NS(FortranRounding), long double);
#endif

NS(ConversionResultFlags)
ConvertDecimalToFloat(const char **, float *, enum NS(FortranRounding));
NS(ConversionResultFlags)
ConvertDecimalToDouble(const char **, double *, enum NS(FortranRounding));
#if __x86_64__
NS(ConversionResultFlags)
ConvertDecimalToLongDouble(
    const char **, long double *, enum NS(FortranRounding));
#endif
#undef NS
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
