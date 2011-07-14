// RUN: %clang_cc1 -verify %s

template<typename T> struct S {
  static int a, b;
};

template<typename T> int S<T>::a, S<T>::b; // expected-error {{can only declare a single entity}}

// FIXME: the last two diagnostics here are terrible.
template<typename T> struct A { static A a; } A<T>::a; // expected-error {{expected ';' after struct}} \
                                                          expected-error {{use of undeclared identifier 'T'}} \
                                                          expected-error {{cannot name the global scope}} \
                                                          expected-error {{no member named 'a' in the global namespace}}

template<typename T> struct B { } f(); // expected-error {{expected ';' after struct}} \
                                          expected-error {{requires a type specifier}}

template<typename T> struct C { } // expected-error {{expected ';' after struct}}

A<int> c;
