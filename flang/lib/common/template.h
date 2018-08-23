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
template<int N, template<typename> class PREDICATE, typename TUPLE>
struct SearchTypeListHelper {
  static constexpr int value() {
    if constexpr (N >= std::tuple_size_v<TUPLE>) {
      return -1;
    } else if constexpr (PREDICATE<std::tuple_element_t<N, TUPLE>>::value()) {
      return N;
    } else {
      return SearchTypeListHelper<N + 1, PREDICATE, TUPLE>::value();
    }
  }
};

template<template<typename> class PREDICATE, typename... TYPES>
constexpr int SearchTypeList{
    SearchTypeListHelper<0, PREDICATE, std::tuple<TYPES...>>::value()};

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
// to match the std::variant type with a partial specialization.  And it
// works with tuples, too, of course.
template<template<typename...> class, typename> struct OverMembersHelper;
template<template<typename...> class T, typename... Ts>
struct OverMembersHelper<T, std::variant<Ts...>> {
  using type = T<Ts...>;
};
template<template<typename...> class T, typename... Ts>
struct OverMembersHelper<T, std::tuple<Ts...>> {
  using type = T<Ts...>;
};

template<template<typename...> class T, typename TorV>
using OverMembers = typename OverMembersHelper<T, TorV>::type;

template<template<typename> class PREDICATE> struct SearchMembersHelper {
  template<typename... Ts> struct Scanner {
    static constexpr int value() { return SearchTypeList<PREDICATE, Ts...>; }
  };
};

template<template<typename> class PREDICATE, typename TorV>
constexpr int SearchMembers{
    OverMembers<SearchMembersHelper<PREDICATE>::template Scanner,
        TorV>::value()};

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
// a new instantiation that holds all of their alternatives, which probably
// should be distinct.
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

// Given a type function, apply it to each of the types in a tuple or variant,
// and collect the results in another tuple or variant.
template<template<typename> class, template<typename...> class, typename...>
struct MapTemplateHelper;
template<template<typename> class F, template<typename...> class TorV,
    typename... Ts>
struct MapTemplateHelper<F, TorV, std::tuple<Ts...>> {
  using type = TorV<F<Ts>...>;
};
template<template<typename> class F, template<typename...> class TorV,
    typename... Ts>
struct MapTemplateHelper<F, TorV, std::variant<Ts...>> {
  using type = TorV<F<Ts>...>;
};
template<template<typename> class F, template<typename...> class TorV,
    typename TV>
using MapTemplate = typename MapTemplateHelper<F, TorV, TV>::type;

}  // namespace Fortran::common
#endif  // FORTRAN_COMMON_TEMPLATE_H_
