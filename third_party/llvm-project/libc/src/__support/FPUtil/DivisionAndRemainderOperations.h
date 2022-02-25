//===-- Floating point divsion and remainder operations ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_DIVISION_AND_REMAINDER_OPERATIONS_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_DIVISION_AND_REMAINDER_OPERATIONS_H

#include "FPBits.h"
#include "ManipulationFunctions.h"
#include "NormalFloat.h"

#include "src/__support/CPP/TypeTraits.h"

namespace __llvm_libc {
namespace fputil {

static constexpr int QUOTIENT_LSB_BITS = 3;

// The implementation is a bit-by-bit algorithm which uses integer division
// to evaluate the quotient and remainder.
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T remquo(T x, T y, int &q) {
  FPBits<T> xbits(x), ybits(y);
  if (xbits.is_nan())
    return x;
  if (ybits.is_nan())
    return y;
  if (xbits.is_inf() || ybits.is_zero())
    return FPBits<T>::build_nan(1);

  if (xbits.is_zero()) {
    q = 0;
    return __llvm_libc::fputil::copysign(T(0.0), x);
  }

  if (ybits.is_inf()) {
    q = 0;
    return x;
  }

  bool result_sign = (xbits.get_sign() == ybits.get_sign() ? false : true);

  // Once we know the sign of the result, we can just operate on the absolute
  // values. The correct sign can be applied to the result after the result
  // is evaluated.
  xbits.set_sign(0);
  ybits.set_sign(0);

  NormalFloat<T> normalx(xbits), normaly(ybits);
  int exp = normalx.exponent - normaly.exponent;
  typename NormalFloat<T>::UIntType mx = normalx.mantissa,
                                    my = normaly.mantissa;

  q = 0;
  while (exp >= 0) {
    unsigned shift_count = 0;
    typename NormalFloat<T>::UIntType n = mx;
    for (shift_count = 0; n < my; n <<= 1, ++shift_count)
      ;

    if (static_cast<int>(shift_count) > exp)
      break;

    exp -= shift_count;
    if (0 <= exp && exp < QUOTIENT_LSB_BITS)
      q |= (1 << exp);

    mx = n - my;
    if (mx == 0) {
      q = result_sign ? -q : q;
      return __llvm_libc::fputil::copysign(T(0.0), x);
    }
  }

  NormalFloat<T> remainder(exp + normaly.exponent, mx, 0);

  // Since NormalFloat to native type conversion is a truncation operation
  // currently, the remainder value in the native type is correct as is.
  // However, if NormalFloat to native type conversion is updated in future,
  // then the conversion to native remainder value should be updated
  // appropriately and some directed tests added.
  T native_remainder(remainder);
  T absy = T(ybits);
  int cmp = remainder.mul2(1).cmp(normaly);
  if (cmp > 0) {
    q = q + 1;
    if (x >= T(0.0))
      native_remainder = native_remainder - absy;
    else
      native_remainder = absy - native_remainder;
  } else if (cmp == 0) {
    if (q & 1) {
      q += 1;
      if (x >= T(0.0))
        native_remainder = -native_remainder;
    } else {
      if (x < T(0.0))
        native_remainder = -native_remainder;
    }
  } else {
    if (x < T(0.0))
      native_remainder = -native_remainder;
  }

  q = result_sign ? -q : q;
  if (native_remainder == T(0.0))
    return __llvm_libc::fputil::copysign(T(0.0), x);
  return native_remainder;
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_DIVISION_AND_REMAINDER_OPERATIONS_H
