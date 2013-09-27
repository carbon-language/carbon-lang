// RUN: %clang_cc1 -std=c++1y -verify %s

class [[deprecated]] C {}; // expected-note {{declared here}}
C c; // expected-warning {{'C' is deprecated}}

typedef int t [[deprecated]]; // expected-note {{declared here}}
t x = 42; // expected-warning {{'t' is deprecated}}

[[deprecated]] int old = 42; // expected-note {{declared here}}
int use = old; // expected-warning {{'old' is deprecated}}

struct S { [[deprecated]] int member = 42; } s; // expected-note {{declared here}}
int use2 = s.member; // expected-warning {{'member' is deprecated}}

[[deprecated]] int f() { return 42; } // expected-note {{declared here}}
int use3 = f(); // expected-warning {{'f' is deprecated}}

enum [[deprecated]] e { E }; // expected-note {{declared here}}
e my_enum; // expected-warning {{'e' is deprecated}}

template <typename T> class X {};
template <> class [[deprecated]] X<int> {}; // expected-note {{declared here}}
X<char> x1;
// FIXME: The diagnostic here could be much better by mentioning X<int>.
X<int> x2; // expected-warning {{'X' is deprecated}}

template <typename T> class [[deprecated]] X2 {};
template <> class X2<int> {};
X2<char> x3; // FIXME: no warning!
X2<int> x4;
