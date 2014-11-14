// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -verify -fexceptions -fcxx-exceptions

// This is a test for an egregious hack in Clang that works around
// an issue with GCC's <utility> implementation. std::pair::swap
// has an exception specification that makes an unqualified call to
// swap. This is invalid, because it ends up calling itself with
// the wrong number of arguments.

#ifdef BE_THE_HEADER

#pragma GCC system_header
namespace std {
  template<typename T> void swap(T &, T &);

  template<typename A, typename B> struct pair {
    void swap(pair &other) noexcept(noexcept(swap(*this, other)));
  };
}

#else

#define BE_THE_HEADER
#include __FILE__

struct X {};
using PX = std::pair<X, X>;
using PI = std::pair<int, int>;
void swap(PX &, PX &) noexcept;
PX px;
PI pi;

static_assert(noexcept(px.swap(px)), "");
static_assert(!noexcept(pi.swap(pi)), "");

namespace sad {
  template<typename T> void swap(T &, T &);

  template<typename A, typename B> struct pair {
    void swap(pair &other) noexcept(noexcept(swap(*this, other))); // expected-error {{too many arguments}} expected-note {{declared here}}
  };

  pair<int, int> pi;

  static_assert(!noexcept(pi.swap(pi)), ""); // expected-note {{in instantiation of}}
}

#endif
