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

#ifndef FORTRAN_EVALUATE_LOGICAL_H_
#define FORTRAN_EVALUATE_LOGICAL_H_

#include "integer.h"
#include <cinttypes>

namespace Fortran::evaluate::value {

template<int BITS> class Logical {
public:
  static constexpr int bits{BITS};
  constexpr Logical() {}  // .FALSE.
  constexpr Logical(const Logical &that) = default;
  constexpr Logical(bool truth) : word_{-std::uint64_t{truth}} {}
  constexpr Logical &operator=(const Logical &) = default;

  // For static expression evaluation, all the bits will have the same value.
  constexpr bool IsTrue() const { return word_.BTEST(0); }

  constexpr Logical NOT() const { return {word_.NOT()}; }

  constexpr Logical AND(const Logical &that) const {
    return {word_.IAND(that.word_)};
  }

  constexpr Logical OR(const Logical &that) const {
    return {word_.IOR(that.word_)};
  }

  constexpr Logical EQV(const Logical &that) const { return NEQV(that).NOT(); }

  constexpr Logical NEQV(const Logical &that) const {
    return {word_.IEOR(that.word_)};
  }

private:
  using Word = Integer<bits>;
  constexpr Logical(const Word &w) : word_{w} {}
  Word word_;
};

extern template class Logical<8>;
extern template class Logical<16>;
extern template class Logical<32>;
extern template class Logical<64>;
}
#endif  // FORTRAN_EVALUATE_LOGICAL_H_
