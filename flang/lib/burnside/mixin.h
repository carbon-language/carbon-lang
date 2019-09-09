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

#ifndef FORTRAN_BURNSIDE_MIXIN_H_
#define FORTRAN_BURNSIDE_MIXIN_H_

// Mixin classes are "partial" classes (not used standalone) that can be used to
// add a repetitive (ad hoc) interface (and implementation) to a class.  It's
// better to think of these as "included in" a class, rather than as an
// "inherited from" base class.

/// [Coding style](https://llvm.org/docs/CodingStandards.html)

#include "llvm/ADT/ilist.h"
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>

namespace Fortran::burnside {

// implementation of a (moveable) sum type (variant)
template<typename... Ts> struct SumTypeMixin {
  using SumTypeTrait = std::true_type;
  template<typename A> SumTypeMixin(const A &x) : u{x} {}
  template<typename A> SumTypeMixin(A &&x) : u{std::forward<A>(x)} {}
  SumTypeMixin(SumTypeMixin &&) = default;
  SumTypeMixin &operator=(SumTypeMixin &&) = default;
  SumTypeMixin(const SumTypeMixin &) = delete;
  SumTypeMixin &operator=(const SumTypeMixin &) = delete;
  SumTypeMixin() = delete;
  std::variant<Ts...> u;
};

// implementation of a copyable sum type
template<typename... Ts> struct SumTypeCopyMixin {
  using CopyableSumTypeTrait = std::true_type;
  template<typename A> SumTypeCopyMixin(const A &x) : u{x} {}
  template<typename A> SumTypeCopyMixin(A &&x) : u{std::forward<A>(x)} {}
  SumTypeCopyMixin(SumTypeCopyMixin &&) = default;
  SumTypeCopyMixin &operator=(SumTypeCopyMixin &&) = default;
  SumTypeCopyMixin(const SumTypeCopyMixin &) = default;
  SumTypeCopyMixin &operator=(const SumTypeCopyMixin &) = default;
  SumTypeCopyMixin() = delete;
  std::variant<Ts...> u;
};
#define SUM_TYPE_COPY_MIXIN(DT) \
  DT(const DT &derived) : SumTypeCopyMixin(derived.u) {} \
  DT(DT &&derived) : SumTypeCopyMixin(std::move(derived.u)) {} \
  DT &operator=(const DT &derived) { \
    SumTypeCopyMixin::operator=(derived.u); \
    return *this; \
  } \
  DT &operator=(DT &&derived) { \
    SumTypeCopyMixin::operator=(std::move(derived.u)); \
    return *this; \
  }

} // namespace burnside

#endif  // FORTRAN_BURNSIDE_MIXIN_H_
