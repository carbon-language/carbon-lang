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

#ifndef FORTRAN_COMMON_KIND_VARIANT_H_
#define FORTRAN_COMMON_KIND_VARIANT_H_

#include "template.h"
#include <utility>
#include <variant>

namespace Fortran::common {

// A KindVariant instantiates a std::variant over a collection of types
// derived by applying a given template to each of a list of "kind" arguments,
// wraps that variant as the sole data member ("u"), and supplies some helpful
// member functions and member function templates to perform reverse
// mappings of both alternative indices and alternative types back to their
// kinds, invoke kind-dependent templates based on dynamic kind values, &c.
template<typename KIND, template<KIND> class TYPE, KIND... KINDS>
struct KindVariant {
  using Kind = KIND;

  static constexpr auto kinds{sizeof...(KINDS)};
  static constexpr Kind kindValue[kinds]{KINDS...};
  template<Kind K> using KindType = TYPE<K>;

  using Variant = std::variant<KindType<KINDS>...>;

  CLASS_BOILERPLATE(KindVariant)
  template<typename A> KindVariant(const A &x) : u{x} {}
  template<typename A>
  KindVariant(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : u{std::move(x)} {}

  template<typename A> KindVariant &operator=(const A &x) {
    u = x;
    return *this;
  }
  template<typename A> KindVariant &operator=(A &&x) {
    u = std::move(x);
    return *this;
  }

  static constexpr Kind IndexToKind(int index) { return kindValue[index]; }

  template<typename A>
  static constexpr Kind TypeToKind{
      IndexToKind(TypeIndex<A, KindType<KINDS>...>)};

  Kind kind() const { return IndexToKind(u.index()); }

  // Accessors for alternatives as identified by kind or type.
  template<Kind K> KindType<K> *GetIfKind() {
    if (auto *p{std::get_if<KindType<K>>(u)}) {
      return p;
    }
    return nullptr;
  }
  template<Kind K> const KindType<K> *GetIfKind() const {
    if (const auto *p{std::get_if<KindType<K>>(u)}) {
      return p;
    }
    return nullptr;
  }
  template<Kind K> std::optional<KindType<K>> GetIf() const {
    return common::GetIf<KindType<K>>(u);
  }

  // Given an instance of some class A with a member template function
  // "template<Kind K> void action();", AtKind<A>(A &a, Kind k) will
  // invoke a.action<k> with a *dynamic* kind value.
private:
  template<typename A, int J> static void Helper(A &a, Kind k) {
    static constexpr Kind K{IndexToKind(J)};
    if (k == K) {
      a.template action<K>();
    } else if constexpr (J + 1 < kinds) {
      Helper<A, J + 1>(a, k);
    }
  }

public:
  template<typename A> static void AtKind(A &a, Kind k) { Helper<A, 0>(a, k); }

  // When each of the alternatives of a KindVariant has a constructor that
  // accepts an rvalue reference to some (same) type A, this template can be
  // used to create a KindVariant instance of a forced kind.
private:
  template<typename A> struct SetResult {
    explicit SetResult(A &&x) : value{std::move(x)} {}
    template<Kind K> void action() {
      CHECK(!result.has_value());
      result = KindVariant{KindType<K>{std::move(value)}};
    }
    std::optional<KindVariant> result;
    A value;
  };

public:
  template<typename A>
  static std::optional<KindVariant> ForceKind(Kind k, A &&x) {
    SetResult<A> setter{std::move(x)};
    AtKind(setter, k);
    return std::move(setter.result);
  }

  Variant u;
};
}  // namespace Fortran::common
#endif  // FORTRAN_COMMON_KIND_VARIANT_H_
