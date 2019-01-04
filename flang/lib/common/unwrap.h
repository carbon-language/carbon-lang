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

#ifndef FORTRAN_COMMON_UNWRAP_H_
#define FORTRAN_COMMON_UNWRAP_H_

#include "indirection.h"
#include "reference-counted.h"
#include <memory>
#include <optional>
#include <type_traits>
#include <variant>

// Given a nest of variants, optionals, &/or pointers, Unwrap<>() isolates
// a packaged value of a specific type if it is present and returns a pointer
// thereto; otherwise, it returns a null pointer.  It's analogous to
// std::get_if<>() but it accepts a reference argument and is recursive.
// The target type parameter cannot be omitted.
//
// Be advised: If the target type parameter is not const-qualified, but the
// isolated value is const-qualified, the result of Unwrap<> will be a
// pointer to a const-qualified value.
//
// Further: const-qualified alternatives in instances of non-const-qualified
// variants will not be returned from Unwrap if the target type is not
// const-qualified.
//
// UnwrapCopy<>() is a variation of Unwrap<>() that returns an optional copy
// of the value if one is present with the desired type.

namespace Fortran::common {

// Utility: Produces "const A" if B is const and A is not already so.
template<typename A, typename B>
using Constify = std::conditional_t<std::is_const_v<B> && !std::is_const_v<A>,
    std::add_const_t<A>, A>;

// Base case
template<typename A, typename B> auto Unwrap(B &x) -> Constify<A, B> * {
  if constexpr (std::is_same_v<std::decay_t<A>, std::decay_t<B>>) {
    return &x;
  } else {
    return nullptr;
  }
}

// Prototypes of specializations, to enable mutual recursion
template<typename A, typename B> auto Unwrap(B *p) -> Constify<A, B> *;
template<typename A, typename B>
auto Unwrap(const std::unique_ptr<B> &) -> Constify<A, B> *;
template<typename A, typename B>
auto Unwrap(const std::shared_ptr<B> &) -> Constify<A, B> *;
template<typename A, typename B>
auto Unwrap(std::optional<B> &) -> Constify<A, B> *;
template<typename A, typename B>
auto Unwrap(const std::optional<B> &) -> std::add_const_t<A> *;
template<typename A, typename... Bs> A *Unwrap(std::variant<Bs...> &);
template<typename A, typename... Bs>
auto Unwrap(const std::variant<Bs...> &) -> std::add_const_t<A> *;
template<typename A, typename B, bool COPY>
auto Unwrap(const Indirection<B, COPY> &) -> Constify<A, B> *;
template<typename A, typename B>
auto Unwrap(const OwningPointer<B> &) -> Constify<A, B> *;
template<typename A, typename B>
auto Unwrap(const CountedReference<B> &) -> Constify<A, B> *;

// Implementations of specializations
template<typename A, typename B> auto Unwrap(B *p) -> Constify<A, B> * {
  if (p != nullptr) {
    return Unwrap<A>(*p);
  } else {
    return nullptr;
  }
}

template<typename A, typename B>
auto Unwrap(const std::unique_ptr<B> &p) -> Constify<A, B> * {
  if (p.get() != nullptr) {
    return Unwrap<A>(*p);
  } else {
    return nullptr;
  }
}

template<typename A, typename B>
auto Unwrap(const std::shared_ptr<B> &p) -> Constify<A, B> * {
  if (p.get() != nullptr) {
    return Unwrap<A>(*p);
  } else {
    return nullptr;
  }
}

template<typename A, typename B>
auto Unwrap(std::optional<B> &x) -> Constify<A, B> * {
  if (x.has_value()) {
    return Unwrap<A>(*x);
  } else {
    return nullptr;
  }
}

template<typename A, typename B>
auto Unwrap(const std::optional<B> &x) -> Constify<A, B> * {
  if (x.has_value()) {
    return Unwrap<A>(*x);
  } else {
    return nullptr;
  }
}

template<typename A, typename... Bs> A *Unwrap(std::variant<Bs...> &u) {
  return std::visit(
      [](auto &x) -> A * {
        using Ty = std::decay_t<decltype(Unwrap<A>(x))>;
        if constexpr (!std::is_const_v<std::remove_pointer_t<Ty>> ||
            std::is_const_v<A>) {
          return Unwrap<A>(x);
        }
        return nullptr;
      },
      u);
}

template<typename A, typename... Bs>
auto Unwrap(const std::variant<Bs...> &u) -> std::add_const_t<A> * {
  return std::visit(
      [](const auto &x) -> std::add_const_t<A> * { return Unwrap<A>(x); }, u);
}

template<typename A, typename B, bool COPY>
auto Unwrap(const Indirection<B, COPY> &p) -> Constify<A, B> * {
  return Unwrap<A>(*p);
}

template<typename A, typename B>
auto Unwrap(const OwningPointer<B> &p) -> Constify<A, B> * {
  if (p.get() != nullptr) {
    return Unwrap<A>(*p);
  } else {
    return nullptr;
  }
}

template<typename A, typename B>
auto Unwrap(const CountedReference<B> &p) -> Constify<A, B> * {
  if (p.get() != nullptr) {
    return Unwrap<A>(*p);
  } else {
    return nullptr;
  }
}

// Returns a copy of a wrapped value, if present, otherwise a vacant optional.
template<typename A, typename B> std::optional<A> UnwrapCopy(const B &x) {
  if (const A * p{Unwrap<A>(x)}) {
    return std::make_optional<A>(*p);
  } else {
    return std::nullopt;
  }
}
}
#endif  // FORTRAN_COMMON_UNWRAP_H_
