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

#ifndef FORTRAN_EVALUATE_INT_POWER_H_
#define FORTRAN_EVALUATE_INT_POWER_H_

// Computes an integer power of a real or complex value.

#include "common.h"

namespace Fortran::evaluate {

template<typename REAL, typename INT>
ValueWithRealFlags<REAL> IntPower(const REAL &base, const INT &power,
    Rounding rounding = Rounding::TiesToEven) {
  REAL one{REAL::FromInteger(INT{1}).value};
  ValueWithRealFlags<REAL> result;
  result.value = one;
  if (base.IsNotANumber()) {
    result.value = REAL::NaN();
    if (base.IsSignalingNaN()) {
      result.flags.set(RealFlag::InvalidArgument);
    }
  } else if (power.IsZero()) {
    if (base.IsZero() || base.IsInfinite()) {
      result.flags.set(RealFlag::InvalidArgument);
    }
  } else {
    bool negativePower{power.IsNegative()};
    INT absPower{power.ABS().value};
    REAL shifted{base};
    int nbits{INT::bits - absPower.LEADZ()};
    for (int j{0}; j + 1 < nbits; ++j) {
      if (absPower.BTEST(j)) {
        result.value =
            result.value.Multiply(shifted).AccumulateFlags(result.flags);
      }
      shifted = shifted.Add(shifted).AccumulateFlags(result.flags);
    }
    result.value = result.value.Multiply(shifted).AccumulateFlags(result.flags);
    if (negativePower) {
      result.value = one.Divide(result.value).AccumulateFlags(result.flags);
    }
  }
  return result;
}

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_INT_POWER_H_
