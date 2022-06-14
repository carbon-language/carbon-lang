//===-- A class to store a normalized floating point number -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_NORMAL_FLOAT_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_NORMAL_FLOAT_H

#include "FPBits.h"

#include "src/__support/CPP/TypeTraits.h"

#include <stdint.h>

namespace __llvm_libc {
namespace fputil {

// A class which stores the normalized form of a floating point value.
// The special IEEE-754 bits patterns of Zero, infinity and NaNs are
// are not handled by this class.
//
// A normalized floating point number is of this form:
//    (-1)*sign * 2^exponent * <mantissa>
// where <mantissa> is of the form 1.<...>.
template <typename T> struct NormalFloat {
  static_assert(
      cpp::IsFloatingPointType<T>::Value,
      "NormalFloat template parameter has to be a floating point type.");

  using UIntType = typename FPBits<T>::UIntType;
  static constexpr UIntType ONE = (UIntType(1) << MantissaWidth<T>::VALUE);

  // Unbiased exponent value.
  int32_t exponent;

  UIntType mantissa;
  // We want |UIntType| to have atleast one bit more than the actual mantissa
  // bit width to accommodate the implicit 1 value.
  static_assert(sizeof(UIntType) * 8 >= MantissaWidth<T>::VALUE + 1,
                "Bad type for mantissa in NormalFloat.");

  bool sign;

  NormalFloat(int32_t e, UIntType m, bool s)
      : exponent(e), mantissa(m), sign(s) {
    if (mantissa >= ONE)
      return;

    unsigned normalization_shift = evaluate_normalization_shift(mantissa);
    mantissa = mantissa << normalization_shift;
    exponent -= normalization_shift;
  }

  explicit NormalFloat(T x) { init_from_bits(FPBits<T>(x)); }

  explicit NormalFloat(FPBits<T> bits) { init_from_bits(bits); }

  // Compares this normalized number with another normalized number.
  // Returns -1 is this number is less than |other|, 0 if this number is equal
  // to |other|, and 1 if this number is greater than |other|.
  int cmp(const NormalFloat<T> &other) const {
    if (sign != other.sign)
      return sign ? -1 : 1;

    if (exponent > other.exponent) {
      return sign ? -1 : 1;
    } else if (exponent == other.exponent) {
      if (mantissa > other.mantissa)
        return sign ? -1 : 1;
      else if (mantissa == other.mantissa)
        return 0;
      else
        return sign ? 1 : -1;
    } else {
      return sign ? 1 : -1;
    }
  }

  // Returns a new normalized floating point number which is equal in value
  // to this number multiplied by 2^e. That is:
  //     new = this *  2^e
  NormalFloat<T> mul2(int e) const {
    NormalFloat<T> result = *this;
    result.exponent += e;
    return result;
  }

  operator T() const {
    int biased_exponent = exponent + FPBits<T>::EXPONENT_BIAS;
    // Max exponent is of the form 0xFF...E. That is why -2 and not -1.
    constexpr int MAX_EXPONENT_VALUE = (1 << ExponentWidth<T>::VALUE) - 2;
    if (biased_exponent > MAX_EXPONENT_VALUE) {
      return sign ? T(FPBits<T>::neg_inf()) : T(FPBits<T>::inf());
    }

    FPBits<T> result(T(0.0));
    result.set_sign(sign);

    constexpr int SUBNORMAL_EXPONENT = -FPBits<T>::EXPONENT_BIAS + 1;
    if (exponent < SUBNORMAL_EXPONENT) {
      unsigned shift = SUBNORMAL_EXPONENT - exponent;
      // Since exponent > subnormalExponent, shift is strictly greater than
      // zero.
      if (shift <= MantissaWidth<T>::VALUE + 1) {
        // Generate a subnormal number. Might lead to loss of precision.
        // We round to nearest and round halfway cases to even.
        const UIntType shift_out_mask = (UIntType(1) << shift) - 1;
        const UIntType shift_out_value = mantissa & shift_out_mask;
        const UIntType halfway_value = UIntType(1) << (shift - 1);
        result.set_unbiased_exponent(0);
        result.set_mantissa(mantissa >> shift);
        UIntType new_mantissa = result.get_mantissa();
        if (shift_out_value > halfway_value) {
          new_mantissa += 1;
        } else if (shift_out_value == halfway_value) {
          // Round to even.
          if (result.get_mantissa() & 0x1)
            new_mantissa += 1;
        }
        result.set_mantissa(new_mantissa);
        // Adding 1 to mantissa can lead to overflow. This can only happen if
        // mantissa was all ones (0b111..11). For such a case, we will carry
        // the overflow into the exponent.
        if (new_mantissa == ONE)
          result.set_unbiased_exponent(1);
        return T(result);
      } else {
        return T(result);
      }
    }

    result.set_unbiased_exponent(exponent + FPBits<T>::EXPONENT_BIAS);
    result.set_mantissa(mantissa);
    return T(result);
  }

private:
  void init_from_bits(FPBits<T> bits) {
    sign = bits.get_sign();

    if (bits.is_inf_or_nan() || bits.is_zero()) {
      // Ignore special bit patterns. Implementations deal with them separately
      // anyway so this should not be a problem.
      exponent = 0;
      mantissa = 0;
      return;
    }

    // Normalize subnormal numbers.
    if (bits.get_unbiased_exponent() == 0) {
      unsigned shift = evaluate_normalization_shift(bits.get_mantissa());
      mantissa = UIntType(bits.get_mantissa()) << shift;
      exponent = 1 - FPBits<T>::EXPONENT_BIAS - shift;
    } else {
      exponent = bits.get_unbiased_exponent() - FPBits<T>::EXPONENT_BIAS;
      mantissa = ONE | bits.get_mantissa();
    }
  }

  unsigned evaluate_normalization_shift(UIntType m) {
    unsigned shift = 0;
    for (; (ONE & m) == 0 && (shift < MantissaWidth<T>::VALUE);
         m <<= 1, ++shift)
      ;
    return shift;
  }
};

#ifdef SPECIAL_X86_LONG_DOUBLE
template <>
inline void NormalFloat<long double>::init_from_bits(FPBits<long double> bits) {
  sign = bits.get_sign();

  if (bits.is_inf_or_nan() || bits.is_zero()) {
    // Ignore special bit patterns. Implementations deal with them separately
    // anyway so this should not be a problem.
    exponent = 0;
    mantissa = 0;
    return;
  }

  if (bits.get_unbiased_exponent() == 0) {
    if (bits.get_implicit_bit() == 0) {
      // Since we ignore zero value, the mantissa in this case is non-zero.
      int normalization_shift =
          evaluate_normalization_shift(bits.get_mantissa());
      exponent = -16382 - normalization_shift;
      mantissa = (bits.get_mantissa() << normalization_shift);
    } else {
      exponent = -16382;
      mantissa = ONE | bits.get_mantissa();
    }
  } else {
    if (bits.get_implicit_bit() == 0) {
      // Invalid number so just store 0 similar to a NaN.
      exponent = 0;
      mantissa = 0;
    } else {
      exponent = bits.get_unbiased_exponent() - 16383;
      mantissa = ONE | bits.get_mantissa();
    }
  }
}

template <> inline NormalFloat<long double>::operator long double() const {
  int biased_exponent = exponent + FPBits<long double>::EXPONENT_BIAS;
  // Max exponent is of the form 0xFF...E. That is why -2 and not -1.
  constexpr int MAX_EXPONENT_VALUE =
      (1 << ExponentWidth<long double>::VALUE) - 2;
  if (biased_exponent > MAX_EXPONENT_VALUE) {
    return sign ? FPBits<long double>::neg_inf() : FPBits<long double>::inf();
  }

  FPBits<long double> result(0.0l);
  result.set_sign(sign);

  constexpr int SUBNORMAL_EXPONENT = -FPBits<long double>::EXPONENT_BIAS + 1;
  if (exponent < SUBNORMAL_EXPONENT) {
    unsigned shift = SUBNORMAL_EXPONENT - exponent;
    if (shift <= MantissaWidth<long double>::VALUE + 1) {
      // Generate a subnormal number. Might lead to loss of precision.
      // We round to nearest and round halfway cases to even.
      const UIntType shift_out_mask = (UIntType(1) << shift) - 1;
      const UIntType shift_out_value = mantissa & shift_out_mask;
      const UIntType halfway_value = UIntType(1) << (shift - 1);
      result.set_unbiased_exponent(0);
      result.set_mantissa(mantissa >> shift);
      UIntType new_mantissa = result.get_mantissa();
      if (shift_out_value > halfway_value) {
        new_mantissa += 1;
      } else if (shift_out_value == halfway_value) {
        // Round to even.
        if (result.get_mantissa() & 0x1)
          new_mantissa += 1;
      }
      result.set_mantissa(new_mantissa);
      // Adding 1 to mantissa can lead to overflow. This can only happen if
      // mantissa was all ones (0b111..11). For such a case, we will carry
      // the overflow into the exponent and set the implicit bit to 1.
      if (new_mantissa == ONE) {
        result.set_unbiased_exponent(1);
        result.set_implicit_bit(1);
      } else {
        result.set_implicit_bit(0);
      }
      return static_cast<long double>(result);
    } else {
      return static_cast<long double>(result);
    }
  }

  result.set_unbiased_exponent(biased_exponent);
  result.set_mantissa(mantissa);
  result.set_implicit_bit(1);
  return static_cast<long double>(result);
}
#endif // SPECIAL_X86_LONG_DOUBLE

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_NORMAL_FLOAT_H
