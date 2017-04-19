// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -fno-modules-error-recovery -I %S/Inputs/template-default-args -std=c++11 %s -DBEGIN= -DEND=
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -fno-modules-error-recovery -I %S/Inputs/template-default-args -std=c++11 %s -DBEGIN="namespace N {" -DEND="}"

BEGIN
template<typename T> struct A;
template<typename T> struct B;
template<typename T> struct C;
template<typename T = int> struct D;
template<typename T = int> struct E {};
template<typename T> struct H {};
template<typename T = int, typename U = int> struct I {};
END

#include "b.h"
#include "d.h"

BEGIN
template<typename T = int> struct A {};
template<typename T> struct B {};
template<typename T = int> struct B;
template<typename T = int> struct C;
template<typename T> struct D {};
template<typename T> struct F {};
template<typename T> struct G {};
template<typename T> struct J {};
template<typename T = int> struct J;
struct K : J<> {};
END

#include "c.h"

BEGIN
A<> a;
B<> b;
extern C<> c;
D<> d;
E<> e;
F<> f;
G<> g; // expected-error {{default argument of 'G' must be imported from module 'X.A' before it is required}}
// expected-note@a.h:7 {{default argument declared here}}
H<> h; // expected-error {{default argument of 'H' must be imported from module 'X.A' before it is required}}
// expected-note@a.h:8 {{default argument declared here}}
I<> i;
L<> *l;
END
