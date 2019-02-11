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

#ifndef FORTRAN_COMMON_IDIOMS_H_
#define FORTRAN_COMMON_IDIOMS_H_

// Defines anything that might ever be useful in more than one source file
// or that is too weird or too specific to the host C++ compiler to be
// exposed elsewhere.

#ifndef __cplusplus
#error this is a C++ program
#endif
#if __cplusplus < 201703L
#error this is a C++17 program
#endif
#if !__clang__ && defined __GNUC__ && __GNUC__ < 7
#error g++ >= 7.2 is required
#endif

#include <functional>
#include <list>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>

#if __GNUC__ == 7
// Avoid a deduction bug in GNU 7.x headers by forcing the answer.
namespace std {
template<typename A>
struct is_trivially_copy_constructible<list<A>> : false_type {};
template<typename A>
struct is_trivially_copy_constructible<optional<list<A>>> : false_type {};
}
#endif

// enable "this is a std::string"s with the 's' suffix
using namespace std::literals::string_literals;

namespace Fortran::common {

// Helper templates for combining a list of lambdas into an anonymous
// struct for use with std::visit() on a std::variant<> sum type.
// E.g.: std::visit(visitors{
//         [&](const firstType &x) { ... },
//         [&](const secondType &x) { ... },
//         ...
//         [&](const auto &catchAll) { ... }}, variantObject);

template<typename... LAMBDAS> struct visitors : LAMBDAS... {
  using LAMBDAS::operator()...;
};

template<typename... LAMBDAS> visitors(LAMBDAS... x)->visitors<LAMBDAS...>;

// Calls std::fprintf(stderr, ...), then abort().
[[noreturn]] void die(const char *, ...);

// For switch statements without default: labels.
#define CRASH_NO_CASE \
  Fortran::common::die("no case at " __FILE__ "(%d)", __LINE__)

// For cheap assertions that should be applied in production.
// To disable, compile with '-DCHECK=(void)'
#ifndef CHECK
#define CHECK(x) \
  ((x) || \
      (Fortran::common::die( \
           "CHECK(" #x ") failed at " __FILE__ "(%d)", __LINE__), \
          false))
#endif

// User-defined type traits that default to false:
// Invoke CLASS_TRAIT(traitName) to define a trait, then put
//   using traitName = std::true_type;  (or false_type)
// into the appropriate class definitions.  You can then use
//   typename std::enable_if_t<traitName<...>, ...>
// in template specialization definitions.
#define CLASS_TRAIT(T) \
  namespace class_trait_ns_##T { \
    template<typename A> std::true_type test(typename A::T *); \
    template<typename A> std::false_type test(...); \
    template<typename A> \
    constexpr bool has_trait{decltype(test<A>(nullptr))::value}; \
    template<typename A> \
    constexpr typename std::enable_if_t<has_trait<A>, bool> trait_value() { \
      using U = typename A::T; \
      return U::value; \
    } \
    template<typename A> \
    constexpr typename std::enable_if_t<!has_trait<A>, bool> trait_value() { \
      return false; \
    } \
  } \
  template<typename A> constexpr bool T{class_trait_ns_##T::trait_value<A>()};

#if !defined ATTRIBUTE_UNUSED && (__clang__ || __GNUC__)
#define ATTRIBUTE_UNUSED __attribute__((unused))
#endif

// Define enum class NAME with the given enumerators, a static
// function EnumToString() that maps enumerators to std::string,
// and a constant NAME_enumSize that captures the number of items
// in the enum class.

std::string EnumIndexToString(int index, const char *names);

template<typename A> struct ListItemCount {
  constexpr ListItemCount(std::initializer_list<A> list) : value{list.size()} {}
  const std::size_t value;
};

#define ENUM_CLASS(NAME, ...) \
  enum class NAME { __VA_ARGS__ }; \
  ATTRIBUTE_UNUSED static constexpr std::size_t NAME##_enumSize{[] { \
    enum { __VA_ARGS__ }; \
    return Fortran::common::ListItemCount{__VA_ARGS__}.value; \
  }()}; \
  ATTRIBUTE_UNUSED static inline std::string EnumToString(NAME e) { \
    return Fortran::common::EnumIndexToString( \
        static_cast<int>(e), #__VA_ARGS__); \
  }

// Given a const reference to a value, return a copy of the value.

template<typename A> A Clone(const A &x) { return x; }
}
#endif  // FORTRAN_COMMON_IDIOMS_H_
