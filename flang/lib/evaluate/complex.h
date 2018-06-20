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
#include <string>

namespace Fortran::evaluate::value {

template<typename REAL_TYPE> class Complex {
public:
  using Part = REAL_TYPE;
  static constexpr int bits{2 * Part::bits};

  constexpr Complex() {}  // (+0.0, +0.0)
  constexpr Complex(const Complex &) = default;
  constexpr Complex(const Part &r, const Part &i) : re_{r}, im_{i} {}
  explicit constexpr Complex(const Part &r) : re_{r} {}

  constexpr const Part &REAL() const { return re_; }
  constexpr const Part &AIMAG() const { return im_; }
  constexpr Complex CONJG() const { return {re_, im_.Negate()}; }
  constexpr Complex Negate() const { return {re_.Negate(), im_.Negate()}; }

  constexpr bool Equals(const Complex &that) const {
    return re_.Compare(that.re_) == Relation::Equal &&
        im_.Compare(that.im_) == Relation::Equal;
  }

  ValueWithRealFlags<Complex> Add(const Complex &) const;
  ValueWithRealFlags<Complex> Subtract(const Complex &) const;
  ValueWithRealFlags<Complex> Multiply(const Complex &) const;
  ValueWithRealFlags<Complex> Divide(const Complex &) const;

  std::string DumpHexadecimal() const;
  // TODO: (C)ABS once Real::HYPOT is done
  // TODO: unit testing

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
