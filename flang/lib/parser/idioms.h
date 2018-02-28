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
#if defined __GNUC__ && __GNUC__ < 7
#error G++ >= 7.0 is required
#endif

#include <list>
#include <optional>
#include <ostream>
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

using namespace std::string_literals;  // enable "this is a std::string"s

namespace Fortran {
namespace parser {

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
}  // namespace parser
}  // namespace Fortran

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

// To make error messages more informative, wrap some type information
// around a false compile-time value, e.g.
//   static_assert(BadType<T>::value, "no case for type");
template<typename A> struct BadType : std::false_type {};

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
    trait_value() { using U = typename A::T; return U::value; } \
  template<typename A> \
    constexpr typename std::enable_if<!has_trait<A>, bool>::type \
    trait_value() { return false; } \
  } \
  template<typename A> constexpr bool T{class_trait_ns_##T::trait_value<A>()}

// Formatting
// TODO: remove when unparser is up and running
namespace Fortran {
namespace parser {
template<typename A>
std::ostream &operator<<(std::ostream &o, const std::optional<A> &x) {
  if (x.has_value()) {
    return o << x.value();
  }
  return o << "()";
}

template<typename A>
std::ostream &operator<<(std::ostream &o, const std::list<A> &xs) {
  if (xs.empty()) {
    return o << "[]";
  }
  char marker{'['};
  for (const auto &x : xs) {
    o << marker << x;
    marker = ' ';
  }
  return o << ']';
}

template<int J, typename T>
std::ostream &formatTuple(std::ostream &o, const T &x) {
  if constexpr (J < std::tuple_size_v<T>) {
    return formatTuple<J + 1>(o << std::get<J>(x), x);
  }
}

template<typename... As>
std::ostream &operator<<(std::ostream &o, const std::tuple<As...> &xs) {
  return formatTuple<0>(o << '{', xs) << '}';
}

template<typename... As>
std::ostream &operator<<(std::ostream &o, const std::variant<As...> &x) {
  return std::visit(
      [&o](const auto &y) -> std::ostream & { return o << y; }, x);
}
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_IDIOMS_H_
