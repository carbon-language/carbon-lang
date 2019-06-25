// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_EVALUATE_FORMATTING_H_
#define FORTRAN_EVALUATE_FORMATTING_H_

// It is inconvenient in C++ to have std::ostream::operator<<() as a direct
// friend function of a class template with many instantiations, so the
// various representational class templates in lib/evaluate format themselves
// via AsFortran(std::ostream &) member functions, which the operator<<()
// overload below will call.  Others have AsFortran() member functions that
// return strings.
//
// This header is meant to be included by the headers that define the several
// representational class templates that need it, not by external clients.

#include "../common/indirection.h"
#include <optional>
#include <ostream>
#include <type_traits>

namespace Fortran::evaluate {

extern bool formatForPGF90;

template<typename A>
auto operator<<(std::ostream &o, const A &x) -> decltype(x.AsFortran(o)) {
  return x.AsFortran(o);
}

template<typename A>
auto operator<<(std::ostream &o, const A &x) -> decltype(o << x.AsFortran()) {
  return o << x.AsFortran();
}

template<typename A, bool COPYABLE>
auto operator<<(
    std::ostream &o, const Fortran::common::Indirection<A, COPYABLE> &x)
    -> decltype(o << x.value()) {
  return o << x.value();
}

template<typename A>
auto operator<<(std::ostream &o, const std::optional<A> &x)
    -> decltype(o << *x) {
  if (x.has_value()) {
    o << *x;
  } else {
    o << "(nullopt)";
  }
  return o;
}
}
#endif  // FORTRAN_EVALUATE_FORMATTING_H_
