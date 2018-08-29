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
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>

// Utility templates for metaprogramming and for composing the
// std::optional<>, std::tuple<>, and std::variant<> containers.

namespace Fortran::common {

// const A * -> std::optional<A>
template<typename A> std::optional<A> GetIfNonNull(const A *p) {
  if (p) {
    return {*p};
  }
  return std::nullopt;
}

// const std::variant<..., A, ...> -> std::optional<A>
// i.e., when a variant holds a value of a particular type, return a copy
// of that value in a std::optional<>.
template<typename A, typename VARIANT>
std::optional<A> GetIf(const VARIANT &u) {
  return GetIfNonNull(std::get_if<A>(&u));
}

// std::optional<std::optional<A>> -> std::optional<A>
template<typename A>
std::optional<A> JoinOptional(std::optional<std::optional<A>> &&x) {
  if (x.has_value()) {
    return std::move(*x);
  }
  return std::nullopt;
}

// Move a value from one variant type to another.  The types allowed in the
// source variant must all be allowed in the destination variant type.
template<typename TOV, typename FROMV> TOV MoveVariant(FROMV &&u) {
  return std::visit(
      [](auto &&x) -> TOV { return {std::move(x)}; }, std::move(u));
}

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

// OverMembers extracts the list of types that constitute the alternatives
// of a std::variant or elements of a std::tuple and passes that list as
// parameter types to a given variadic template.
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

// SearchMembers<PREDICATE> scans the types that constitute the alternatives
// of a std::variant instantiation or elements of a std::tuple.
// The zero-based index of the first type T among the alternatives for which
// PREDICATE<T>::value() is true is returned, or -1 when the predicate is false
// for every type in the set.
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
// a new instantiation that holds all of their alternatives, which must be
// pairwise distinct.
template<typename> struct VariantToTupleHelper;
template<typename... Ts> struct VariantToTupleHelper<std::variant<Ts...>> {
  using type = std::tuple<Ts...>;
};
template<typename VARIANT>
using VariantToTuple = typename VariantToTupleHelper<VARIANT>::type;

template<typename A, typename B, typename... REST>
struct AreTypesDistinctHelper {
  static constexpr bool value() {
    if constexpr (std::is_same_v<A, B>) {
      return false;
    }
    if constexpr (sizeof...(REST) > 0) {
      return AreTypesDistinctHelper<A, REST...>::value() &&
          AreTypesDistinctHelper<B, REST...>::value();
    }
    return true;
  }
};
template<typename... Ts>
constexpr bool AreTypesDistinct{AreTypesDistinctHelper<Ts...>::value()};

template<typename> struct TupleToVariantHelper;
template<typename... Ts> struct TupleToVariantHelper<std::tuple<Ts...>> {
  static_assert(AreTypesDistinct<Ts...> ||
      !"TupleToVariant: types are not pairwise distinct");
  using type = std::variant<Ts...>;
};
template<typename TUPLE>
using TupleToVariant = typename TupleToVariantHelper<TUPLE>::type;

template<typename... VARIANTS> struct CombineVariantsHelper {
  using type = TupleToVariant<CombineTuples<VariantToTuple<VARIANTS>...>>;
};
template<typename... VARIANTS>
using CombineVariants = typename CombineVariantsHelper<VARIANTS...>::type;

// SquashVariantOfVariants: given a std::variant whose alternatives are
// all std::variant instantiations, form a new union over their alternatives.
template<typename VARIANT>
using SquashVariantOfVariants = OverMembers<CombineVariants, VARIANT>;

// Given a type function, MapTemplate applies it to each of the types
// in a tuple or variant, and collect the results in a given variadic
// template (typically a std::variant).
template<template<typename> class, template<typename...> class, typename...>
struct MapTemplateHelper;
template<template<typename> class F, template<typename...> class PACKAGE,
    typename... Ts>
struct MapTemplateHelper<F, PACKAGE, std::tuple<Ts...>> {
  using type = PACKAGE<F<Ts>...>;
};
template<template<typename> class F, template<typename...> class PACKAGE,
    typename... Ts>
struct MapTemplateHelper<F, PACKAGE, std::variant<Ts...>> {
  using type = PACKAGE<F<Ts>...>;
};
template<template<typename> class F, typename TorV,
    template<typename...> class PACKAGE = std::variant>
using MapTemplate = typename MapTemplateHelper<F, PACKAGE, TorV>::type;

// std::tuple<std::optional<>...> -> std::optional<std::tuple<...>>
// i.e., inverts a tuple of optional values into an optional tuple that has
// a value only if all of the original elements were present.
template<typename... A, std::size_t... J>
std::optional<std::tuple<A...>> AllElementsPresentHelper(
    std::tuple<std::optional<A>...> &&t, std::index_sequence<J...>) {
  bool present[]{std::get<J>(t).has_value()...};
  for (std::size_t j{0}; j < sizeof...(J); ++j) {
    if (!present[j]) {
      return std::nullopt;
    }
  }
  return {std::make_tuple(*std::get<J>(t)...)};
}

template<typename... A>
std::optional<std::tuple<A...>> AllElementsPresent(
    std::tuple<std::optional<A>...> &&t) {
  return AllElementsPresentHelper(
      std::move(t), std::index_sequence_for<A...>{});
}

// (std::optional<>...) -> std::optional<std::tuple<...>>
// i.e., given some number of optional values, return a optional tuple of
// those values that is present only of all of the values were so.
template<typename... A>
std::optional<std::tuple<A...>> AllPresent(std::optional<A> &&... x) {
  return AllElementsPresent(std::make_tuple(std::move(x)...));
}

// (f(A...) -> R) -> std::optional<A>... -> std::optional<R>
// Apply a function to optional arguments if all are present.
// If the function returns std::optional, you will probably want to
// pass it through JoinOptional to "squash" it.
template<typename R, typename... A>
std::optional<R> MapOptional(
    std::function<R(A &&...)> &&f, std::optional<A> &&... x) {
  if (auto args{AllPresent(std::move(x)...)}) {
    return std::make_optional(std::apply(std::move(f), std::move(*args)));
  }
  return std::nullopt;
}

}  // namespace Fortran::common
#endif  // FORTRAN_COMMON_TEMPLATE_H_
