#ifndef FORTRAN_IDIOMS_H_
#define FORTRAN_IDIOMS_H_

// Defines anything that might ever be useful in more than one source file
// or that is too weird or too specific to the host C++ compiler to be
// exposed elsewhere.

#ifndef __cplusplus
# error this is a C++ program
#endif
#if __cplusplus < 201703L
# error this is a C++17 program
#endif
#if defined __GNUC__ && __GNUC__ < 7
# error G++ >= 7.0 is required
#endif

#include <list>
#include <optional>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <variant>

// Avoid a deduction bug in GNU 7.1.0 headers by forcing the answer.
// TODO: better resolution
namespace std {
template<typename A>
struct is_trivially_copy_constructible<list<A>> : false_type {};
template<typename A>
struct is_trivially_copy_constructible<optional<list<A>>> : false_type {};
}  // namespace std

using namespace std::string_literals;  // enable "this is a std::string"s

namespace Fortran {

// Helper templates for combining a list of lambdas into an anonymous
// struct for use with std::visit() on a std::variant<> sum type.
// E.g.: std::visit(visitors{
//         [&](const UnaryExpr &x) { ... },
//         [&](const BinaryExpr &x) { ... },
//         ...
//       }, structure.unionMember);

template<typename... LAMBDAS>
struct visitors : LAMBDAS... {
  using LAMBDAS::operator()...;
};

template<typename... LAMBDAS>
visitors(LAMBDAS... x) -> visitors<LAMBDAS...>;

// Calls std::fprintf(stderr, ...), then abort().
void die(const char *, ...);

// Treat operator! as if it were a Boolean context, i.e. like if() and ? :,
// when its operand is std::optional<>.
template<typename A> bool operator!(const std::optional<A> &x) {
  return !x.has_value();
}
}  // namespace Fortran

// For switch statements without default: labels.
#define DEFAULT_CRASH \
  default: die("no case at " __FILE__ "(%d)", __LINE__)

// For cheap assertions that should be applied in production.
#define CHECK(x) \
  ((x) || (die("CHECK(" #x ") failed at " __FILE__ "(%d)", __LINE__), false))

// To make error messages more informative, wrap some type information
// around a false compile-time value, e.g.
//   static_assert(BadType<T>::value, "no case for type");
template<typename A> struct BadType : std::false_type {};

// Formatting
namespace Fortran {
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

template<int J, char C, typename T>
typename std::enable_if<J+1 == std::tuple_size_v<T>, std::ostream &>::type
formatTuple(std::ostream &o, const T &x) {
  return o << C << std::get<J>(x) << '}';
}

template<int J, char C, typename T>
typename std::enable_if<J+1 != std::tuple_size_v<T>, std::ostream &>::type
formatTuple(std::ostream &o, const T &x) {
  return formatTuple<J+1,' '>(o << C << std::get<J>(x), x);
}

template<typename... As>
std::ostream &operator<<(std::ostream &o, const std::tuple<As...> &xs) {
  return formatTuple<0, '{'>(o, xs);
}

template<typename... As>
std::ostream &operator<<(std::ostream &o, const std::variant<As...> &x) {
  return std::visit([&o](const auto &y)->std::ostream &{ return o << y; }, x);
}
}  // namespace Fortran
#endif  // FORTRAN_IDIOMS_H_
