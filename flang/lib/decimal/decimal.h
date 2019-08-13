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

enum DecimalConversionFlags {
  RoundToNearestEven = 1,
  AlwaysSign = 2, /* emit leading '+' if not negative */
};

enum BinaryConversionFlags {
  Overflow = 1,
  Inexact = 2,
  Invalid = 4,
};

#ifdef __cplusplus
namespace Fortran::decimal {

template<int PREC>
ConversionToDecimalResult ConvertToDecimal(char *, size_t, int flags,
    int digits, bool minimal, BinaryFloatingPointNumber<PREC> x);

extern template ConversionToDecimalResult ConvertToDecimal<8>(
    char *, size_t, int, int, bool, BinaryFloatingPointNumber<8>);
extern template ConversionToDecimalResult ConvertToDecimal<11>(
    char *, size_t, int, int, bool, BinaryFloatingPointNumber<11>);
extern template ConversionToDecimalResult ConvertToDecimal<24>(
    char *, size_t, int, int, bool, BinaryFloatingPointNumber<24>);
extern template ConversionToDecimalResult ConvertToDecimal<53>(
    char *, size_t, int, int, bool, BinaryFloatingPointNumber<53>);
extern template ConversionToDecimalResult ConvertToDecimal<64>(
    char *, size_t, int, int, bool, BinaryFloatingPointNumber<64>);
extern template ConversionToDecimalResult ConvertToDecimal<112>(
    char *, size_t, int, int, bool, BinaryFloatingPointNumber<112>);

template<int PREC> struct ConversionToBinaryResult {
  BinaryFloatingPointNumber<PREC> binary;
  enum BinaryConversionFlags flags;
};

template<int PREC>
ConversionToBinaryResult<PREC> ConvertToBinary(
    const char *&, bool rounding = true);

extern template ConversionToBinaryResult<8> ConvertToBinary<8>(
    const char *&, bool rounding = true);
extern template ConversionToBinaryResult<11> ConvertToBinary<11>(
    const char *&, bool rounding = true);
extern template ConversionToBinaryResult<24> ConvertToBinary<24>(
    const char *&, bool rounding = true);
extern template ConversionToBinaryResult<53> ConvertToBinary<53>(
    const char *&, bool rounding = true);
extern template ConversionToBinaryResult<64> ConvertToBinary<64>(
    const char *&, bool rounding = true);
extern template ConversionToBinaryResult<112> ConvertToBinary<112>(
    const char *&, bool rounding = true);
}  // namespace
extern "C" {
#endif /* C++ */

ConversionToDecimalResult ConvertFloatToDecimal(
    char *, size_t, int flags, int digits, bool minimal, float);
ConversionToDecimalResult ConvertDoubleToDecimal(
    char *, size_t, int flags, int digits, bool minimal, double);
#if __x86_64__
ConversionToDecimalResult ConvertLongDoubleToDecimal(
    char *, size_t, int flags, int digits, bool minimal, long double);
#endif

#ifdef __cplusplus
}
#endif /* C++ */
#endif /* DECIMAL_H_ */
