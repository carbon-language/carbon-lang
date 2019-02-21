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

#ifndef FORTRAN_INTERMEDIATEREPRESENTATION_MIXIN_H_
#define FORTRAN_INTERMEDIATEREPRESENTATION_MIXIN_H_

#include "llvm/ADT/ilist.h"
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>

namespace Fortran::IntermediateRepresentation {

template<typename T, typename E = void> struct SumTypeMixin {};
template<typename T>  // T must be std::variant<...>
struct SumTypeMixin<T, std::enable_if_t<std::variant_size_v<T>>> {
  template<typename A> SumTypeMixin(A &&x) : u{std::move(x)} {}
  using SumTypeTrait = std::true_type;
  SumTypeMixin(SumTypeMixin &&) = default;
  SumTypeMixin &operator=(SumTypeMixin &&) = default;
  SumTypeMixin(const SumTypeMixin &) = delete;
  SumTypeMixin &operator=(const SumTypeMixin &) = delete;
  SumTypeMixin() = delete;
  T u;
};

template<typename T, typename E = void> struct SumTypeCopyMixin {};
template<typename T>  // T must be std::variant<...>
struct SumTypeCopyMixin<T, std::enable_if_t<std::variant_size_v<T>>> {
  using CopyableSumTypeTrait = std::true_type;
  SumTypeCopyMixin(SumTypeCopyMixin &&) = default;
  SumTypeCopyMixin &operator=(SumTypeCopyMixin &&) = default;
  SumTypeCopyMixin(const SumTypeCopyMixin &) = default;
  SumTypeCopyMixin &operator=(const SumTypeCopyMixin &) = default;
  SumTypeCopyMixin() = delete;
  T u;
};
#define SUM_TYPE_COPY_MIXIN(Derived) \
  Derived(const Derived &derived) : SumTypeCopyMixin(derived) {} \
  Derived &operator=(const Derived &derived) { \
    SumTypeCopyMixin::operator=(derived); \
    return *this; \
  }

template<typename T, typename E = void> struct ProductTypeMixin {};
template<typename T>  // T must be std::tuple<...>
struct ProductTypeMixin<T, std::enable_if_t<std::tuple_size_v<T>>> {
  template<typename A> ProductTypeMixin(A &&x) : t{std::move(x)} {}
  using ProductTypeTrait = std::true_type;
  ProductTypeMixin(ProductTypeMixin &&) = default;
  ProductTypeMixin &operator=(ProductTypeMixin &&) = default;
  ProductTypeMixin(const ProductTypeMixin &) = delete;
  ProductTypeMixin &operator=(const ProductTypeMixin &) = delete;
  ProductTypeMixin() = delete;
  T t;
};

template<typename T, typename E = void> struct MaybeMixin {};
template<typename T>  // T must be std::optional<...>
struct MaybeMixin<T,
    std::enable_if_t<
        std::is_same_v<std::optional<typename T::value_type>, T>>> {
  template<typename A> MaybeMixin(A &&x) : o{std::move(x)} {}
  using MaybeTrait = std::true_type;
  MaybeMixin(MaybeMixin &&) = default;
  MaybeMixin &operator=(MaybeMixin &&) = default;
  MaybeMixin(const MaybeMixin &) = delete;
  MaybeMixin &operator=(const MaybeMixin &) = delete;
  MaybeMixin() = delete;
  T o;
};

template<typename T, typename P> struct ChildMixin {
protected:
  P *parent;

public:
  ChildMixin(P *p) : parent{p} {}
  inline const P *getParent() const { return parent; }
  inline P *getParent() { return parent; }
  llvm::iplist<T> &getList() { return parent->getSublist(this); }
};

template<typename A, typename B, typename C>
C Zip(C out, A first, A last, B other) {
  std::transform(first, last, other, out,
      [](auto &&a, auto &&b) -> std::pair<decltype(a), decltype(b)> {
        return {a, b};
      });
  return out;
}
template<typename A, typename B> B &Unzip(B &out, A first, A last) {
  std::transform(first, last, std::back_inserter(out.first),
      [](auto &&a) -> decltype(a.first) { return a.first; });
  std::transform(first, last, std::back_inserter(out.second),
      [](auto &&a) -> decltype(a.second) { return a.second; });
  return out;
}

}

#endif  // FORTRAN_INTERMEDIATEREPRESENTATION_COMMON_H_
