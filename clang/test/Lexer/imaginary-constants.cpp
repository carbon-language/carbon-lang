// RUN: %clang_cc1 -fsyntax-only -verify %s -std=gnu++98 -DHAVE_IMAGINARY=1
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=gnu++11 -DHAVE_IMAGINARY=1
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=gnu++14 -DHAVE_IMAGINARY=1
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++98 -DHAVE_IMAGINARY=0
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11 -DHAVE_IMAGINARY=0
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++14 -DHAVE_IMAGINARY=0 -DCXX14=1

// Imaginary constants are a GNU extension that became problematic when C++14
// defined its own versions. Until then they're supported even in
// standards-compliant mode.
#if HAVE_IMAGINARY
// expected-no-diagnostics
#elif CXX14
// expected-error@+9 {{no matching literal operator for call to 'operator""i' with argument of type 'unsigned long long' or 'const char *', and no matching literal operator template}}
// expected-error@+9 {{no matching literal operator for call to 'operator""il' with argument of type 'unsigned long long' or 'const char *', and no matching literal operator template}}
// expected-error@+9 {{invalid suffix 'ill' on integer constant}}
#else
// expected-error@+5 {{invalid suffix 'i' on integer constant}}
// expected-error@+5 {{invalid suffix 'il' on integer constant}}
// expected-error@+7 {{invalid suffix 'ill' on integer constant}}
#endif

_Complex int val1 = 2i;
_Complex long val2 = 2il;
_Complex long long val3 = 2ill;
