// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify -include %s %s

// libstdc++4.6 has a broken numeric_limits with a non-constant min() for signed
// integral types. Disable the 'never produces a constant expression' error in
// system headers to work around it. We still won't treat the function as
// producing a constant expression, though.

#ifndef INCLUDED_HEADER
#define INCLUDED_HEADER

#pragma GCC system_header

// An approximation of libstdc++4.6's broken definition of numeric_limits.
// FIXME: In the -include case, the line numbers are off by one for some reason!
struct numeric_limits { // expected-note {{value 2147483648 is outside the range}}
  static constexpr int min() throw() { return (int)1 << (sizeof(int) * 8 - 1); } // no-error
  // expected-note {{in call to 'min()'}}
  static constexpr int lowest() throw() { return min(); }
};

#else

constexpr int k = numeric_limits::lowest(); // expected-error {{constant expression}} expected-note {{in call to 'lowest()'}}

#endif
