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

#ifndef FORTRAN_PARSER_IDIOMS_H_
#define FORTRAN_PARSER_IDIOMS_H_

// Defines anything that might ever be useful in more than one source file
// or that is too weird or too specific to the host C++ compiler to be
// exposed elsewhere.

#ifndef __cplusplus
#error this is a C++ program
#endif
#if __cplusplus < 201703L
#error this is a C++17 program
#endif
#if !defined(__clang__) && defined __GNUC__ && __GNUC__ < 7
#error G++ >= 7.0 is required
#endif

#include <list>
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>

// Avoid a deduction bug in GNU 7.3.0 headers by forcing the answer.
// TODO: better resolution
namespace std {
template<typename A>
struct is_trivially_copy_constructible<list<A>> : false_type {};
template<typename A>
struct is_trivially_copy_constructible<optional<list<A>>> : false_type {};
}  // namespace std

// enable "this is a std::string"s with the 's' suffix
using namespace std::literals::string_literals;

namespace Fortran::parser {

// Helper templates for combining a list of lambdas into an anonymous
// struct for use with std::visit() on a std::variant<> sum type.
// E.g.: std::visit(visitors{
//         [&](const UnaryExpr &x) { ... },
//         [&](const BinaryExpr &x) { ... },
//         ...
//       }, structure.unionMember);

template<typename... LAMBDAS> struct visitors : LAMBDAS... {
  using LAMBDAS::operator()...;
};

template<typename... LAMBDAS> visitors(LAMBDAS... x)->visitors<LAMBDAS...>;

// Calls std::fprintf(stderr, ...), then abort().
[[noreturn]] void die(const char *, ...);

// Treat operator! as if it were a Boolean context, i.e. like if() and ? :,
// when its operand is std::optional<>.
template<typename A> bool operator!(const std::optional<A> &x) {
  return !x.has_value();
}

// For switch statements without default: labels.
#define CRASH_NO_CASE \
  Fortran::parser::die("no case at " __FILE__ "(%d)", __LINE__)

// For cheap assertions that should be applied in production.
// To disable, compile with '-DCHECK=(void)'
#ifndef CHECK
#define CHECK(x) \
  ((x) || \
      (Fortran::parser::die( \
           "CHECK(" #x ") failed at " __FILE__ "(%d)", __LINE__), \
          false))
#endif

// User-defined type traits that default to false:
// Invoke CLASS_TRAIT(traitName) to define a trait, then put
//   using traitName = std::true_type;  (or false_type)
// into the appropriate class definitions.  You can then use
//   typename std::enable_if<traitName<...>, ...>::type
// in template specialization definitions.
#define CLASS_TRAIT(T) \
  namespace class_trait_ns_##T { \
    template<typename A> std::true_type test(typename A::T *); \
    template<typename A> std::false_type test(...); \
    template<typename A> \
    constexpr bool has_trait{decltype(test<A>(nullptr))::value}; \
    template<typename A> \
    constexpr typename std::enable_if<has_trait<A>, bool>::type \
    trait_value() { \
      using U = typename A::T; \
      return U::value; \
    } \
    template<typename A> \
    constexpr typename std::enable_if<!has_trait<A>, bool>::type \
    trait_value() { \
      return false; \
    } \
  } \
  template<typename A> constexpr bool T { class_trait_ns_##T::trait_value<A>() }

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
  static constexpr std::size_t NAME##_enumSize{[] { \
    enum { __VA_ARGS__ }; \
    return Fortran::parser::ListItemCount{__VA_ARGS__}.value; \
  }()}; \
  static inline std::string EnumToString(NAME e) { \
    return Fortran::parser::EnumIndexToString( \
        static_cast<int>(e), #__VA_ARGS__); \
  }

}  // namespace Fortran::parser
#endif  // FORTRAN_PARSER_IDIOMS_H_
