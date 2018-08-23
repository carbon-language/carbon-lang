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

#ifndef FORTRAN_COMMON_TEMPLATE_H_
#define FORTRAN_COMMON_TEMPLATE_H_

#include <functional>
#include <tuple>
#include <type_traits>
#include <variant>

// Template metaprogramming utilities

namespace Fortran::common {

// SearchTypeList<PREDICATE, TYPES...> scans a list of types.  The zero-based
// index of the first type T in the list for which PREDICATE<T>::value() is
// true is returned, or -1 if the predicate is false for every type in the list.
template<int N, template<typename> class PREDICATE, typename A,
    typename... REST>
struct SearchTypeListTemplate {
  static constexpr int value() {
    if constexpr (PREDICATE<A>::value()) {
      return N;
    } else if constexpr (sizeof...(REST) == 0) {
      return -1;
    } else {
      return SearchTypeListTemplate<N + 1, PREDICATE, REST...>::value();
    }
  }
};

template<template<typename> class PREDICATE, typename... TYPES>
constexpr int SearchTypeList{
    SearchTypeListTemplate<0, PREDICATE, TYPES...>::value()};

// TypeIndex<A, TYPES...> scans a list of types for simple type equality.
// The zero-based index of A in the list is returned, or -1 if A is not present.
template<typename A> struct MatchType {
  template<typename B> struct Match {
    static constexpr bool value() {
      return std::is_same_v<std::decay_t<A>, std::decay_t<B>>;
    }
  };
};

template<typename A, typename... TYPES>
constexpr int TypeIndex{SearchTypeList<MatchType<A>::template Match, TYPES...>};

// SearchVariantType<PREDICATE> scans the types that constitute the alternatives
// of a std::variant instantiation.  The zero-based index of the first type T
// among the alternatives for which PREDICATE<T>::value() is true is returned,
// or -1 if the predicate is false for every alternative of the union.

// N.B. It *is* possible to extract the types of the alternatives of a
// std::variant discriminated union instantiation and reuse them as a
// template parameter pack in another template instantiation.  The trick is
// to match the std::variant type with a partial specialization.
template<template<typename> class PREDICATE, typename V>
struct SearchVariantTypeTemplate;
template<template<typename> class PREDICATE, typename... Ts>
struct SearchVariantTypeTemplate<PREDICATE, std::variant<Ts...>> {
  static constexpr int index{SearchTypeList<PREDICATE, Ts...>};
};

template<template<typename> class PREDICATE, typename VARIANT>
constexpr int SearchVariantType{
    SearchVariantTypeTemplate<PREDICATE, VARIANT>::index};

// CombineTuples takes a list of std::tuple<> template instantiation types
// and constructs a new std::tuple type that concatenates all of their member
// types.  E.g.,
//   CombineTuples<std::tuple<char, int>, std::tuple<float, double>>
// is std::tuple<char, int, float, double>.
template<typename... TUPLES> struct CombineTuplesHelper {
  static decltype(auto) f(TUPLES *... a) {
    return std::tuple_cat(std::move(*a)...);
  }
  using type = decltype(f(static_cast<TUPLES *>(nullptr)...));
};
template<typename... TUPLES>
using CombineTuples = typename CombineTuplesHelper<TUPLES...>::type;

// CombineVariants takes a list of std::variant<> instantiations and constructs
// a new instantiation that holds all of their alternatives.
template<typename> struct VariantToTupleHelper;
template<typename... Ts> struct VariantToTupleHelper<std::variant<Ts...>> {
  using type = std::tuple<Ts...>;
};
template<typename VARIANT>
using VariantToTuple = typename VariantToTupleHelper<VARIANT>::type;

template<typename> struct TupleToVariantHelper;
template<typename... Ts> struct TupleToVariantHelper<std::tuple<Ts...>> {
  using type = std::variant<Ts...>;
};
template<typename TUPLE>
using TupleToVariant = typename TupleToVariantHelper<TUPLE>::type;

template<typename... VARIANTS> struct CombineVariantsHelper {
  using type = TupleToVariant<CombineTuples<VariantToTuple<VARIANTS>...>>;
};
template<typename... VARIANTS>
using CombineVariants = typename CombineVariantsHelper<VARIANTS...>::type;

}  // namespace Fortran::common
#endif  // FORTRAN_COMMON_TEMPLATE_H_
