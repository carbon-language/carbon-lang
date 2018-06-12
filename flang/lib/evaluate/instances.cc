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

#include "complex.h"
#include "integer.h"
#include "logical.h"
#include "real.h"

namespace Fortran::evaluate::value {

template class Integer<8>;
template class Integer<16>;
template class Integer<32>;
template class Integer<64>;
template class Integer<80>;
template class Integer<128>;

template class Real<Integer<16>, 11>;
template class Real<Integer<32>, 24>;
template class Real<Integer<64>, 53>;
template class Real<Integer<80>, 64, false>;
template class Real<Integer<128>, 112>;

template class Complex<Real<Integer<16>, 11>>;
template class Complex<Real<Integer<32>, 24>>;
template class Complex<Real<Integer<64>, 53>>;
template class Complex<Real<Integer<80>, 64, false>>;
template class Complex<Real<Integer<128>, 112>>;

template class Logical<8>;
template class Logical<16>;
template class Logical<32>;
template class Logical<64>;
template class Logical<128>;

// Sanity checks against misconfiguration bugs
static_assert(Integer<8>::partBits == 8);
static_assert(std::is_same_v<typename Integer<8>::Part, std::uint8_t>);
static_assert(Integer<16>::partBits == 16);
static_assert(std::is_same_v<typename Integer<16>::Part, std::uint16_t>);
static_assert(Integer<32>::partBits == 32);
static_assert(std::is_same_v<typename Integer<32>::Part, std::uint32_t>);
static_assert(Integer<64>::partBits == 32);
static_assert(std::is_same_v<typename Integer<64>::Part, std::uint32_t>);
static_assert(Integer<128>::partBits == 32);
static_assert(std::is_same_v<typename Integer<128>::Part, std::uint32_t>);

}  // namespace Fortran::evaluate::value
