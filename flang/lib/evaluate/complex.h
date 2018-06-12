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

#ifndef FORTRAN_EVALUATE_COMPLEX_H_
#define FORTRAN_EVALUATE_COMPLEX_H_

#include "real.h"

namespace Fortran::evaluate::value {

template<typename REAL_TYPE> class Complex {
public:
  using Part = REAL_TYPE;
  static constexpr int bits{2 * Part::bits};

  constexpr Complex() {}  // (+0.0, +0.0)
  constexpr Complex(const Complex &) = default;
  constexpr Complex(const Part &r, const Part &i) : re_{r}, im_{i} {}
  explicit constexpr Complex(const Part &r) : re_{r} {}

  // TODO: (C)ABS, unit testing

  constexpr const Part &REAL() const { return re_; }
  constexpr const Part &AIMAG() const { return im_; }
  constexpr Complex CONJG() const { return {re_, im_.Negate()}; }
  constexpr Complex Negate() const { return {re_.Negate(), im_.Negate()}; }

  constexpr bool Equals(const Complex &that) const {
    return re_.Compare(that.re_) == Relation::Equal &&
        im_.Compare(that.im_) == Relation::Equal;
  }

  constexpr ValueWithRealFlags<Complex> Add(const Complex &that) const {
    RealFlags flags;
    Part reSum{re_.Add(that.re_).AccumulateFlags(flags)};
    Part imSum{im_.Add(that.im_).AccumulateFlags(flags)};
    return {Complex{reSum, imSum}, flags};
  }

  constexpr ValueWithRealFlags<Complex> Subtract(const Complex &that) const {
    RealFlags flags;
    Part reDiff{re_.Subtract(that.re_).AccumulateFlags(flags)};
    Part imDiff{im_.Subtract(that.im_).AccumulateFlags(flags)};
    return {Complex{reDiff, imDiff}, flags};
  }

  constexpr ValueWithRealFlags<Complex> Multiply(const Complex &that) const {
    // (a + ib)*(c + id) -> ac - bd + i(ad + bc)
    RealFlags flags;
    Part ac{re_.Multiply(that.re_).AccumulateFlags(flags)};
    Part bd{im_.Multiply(that.im_).AccumulateFlags(flags)};
    Part ad{re_.Multiply(that.im_).AccumulateFlags(flags)};
    Part bc{im_.Multiply(that.re_).AccumulateFlags(flags)};
    Part acbd{ac.Subtract(bd).AccumulateFlags(flags)};
    Part adbc{ad.Add(bc).AccumulateFlags(flags)};
    return {Complex{acbd, adbc}, flags};
  }

  constexpr ValueWithRealFlags<Complex> Divide(const Complex &that) const {
    // (a + ib)/(c + id) -> [(a+ib)*(c-id)] / [(c+id)*(c-id)]
    //   -> [ac+bd+i(bc-ad)] / (cc+dd)
    //   -> ((ac+bd)/(cc+dd)) + i((bc-ad)/(cc+dd))
    // but to avoid overflows, scale by d/c if c>=d, else c/d
    Part scale;  // <= 1.0
    RealFlags flags;
    bool cGEd{that.re_.ABS().Compare(that.im_.ABS()) != Relation::Less};
    if (cGEd) {
      scale = that.im_.Divide(that.re_).AccumulateFlags(flags);
    } else {
      scale = that.re_.Divide(that.im_).AccumulateFlags(flags);
    }
    Part den;
    if (cGEd) {
      Part dS{scale.Multiply(that.im_).AccumulateFlags(flags)};
      den = dS.Add(that.re_).AccumulateFlags(flags);
    } else {
      Part cS{scale.Multiply(that.re_).AccumulateFlags(flags)};
      den = cS.Add(that.im_).AccumulateFlags(flags);
    }
    Part aS{scale.Multiply(re_).AccumulateFlags(flags)};
    Part bS{scale.Multiply(im_).AccumulateFlags(flags)};
    Part re1, im1;
    if (cGEd) {
      re1 = re_.Add(bS).AccumulateFlags(flags);
      im1 = im_.Subtract(aS).AccumulateFlags(flags);
    } else {
      re1 = aS.Add(im_).AccumulateFlags(flags);
      im1 = bS.Subtract(re_).AccumulateFlags(flags);
    }
    Part re{re1.Divide(den).AccumulateFlags(flags)};
    Part im{im1.Divide(den).AccumulateFlags(flags)};
    return {Complex{re, im}, flags};
  }

private:
  Part re_, im_;
};

extern template class Complex<Real<Integer<16>, 11>>;
extern template class Complex<Real<Integer<32>, 24>>;
extern template class Complex<Real<Integer<64>, 53>>;
extern template class Complex<Real<Integer<80>, 64, false>>;
extern template class Complex<Real<Integer<128>, 112>>;
}  // namespace Fortran::evaluate::value
#endif  // FORTRAN_EVALUATE_COMPLEX_H_
