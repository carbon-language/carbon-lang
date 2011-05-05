// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

template<typename T> using A = int;
template<typename T> using A<T*> = char; // expected-error {{partial specialization of alias templates is not permitted}}
template<> using A<char> = char; // expected-error {{explicit specialization of alias templates is not permitted}}
template using A<char> = char; // expected-error {{explicit instantiation of alias templates is not permitted}}
// Best guess as to what the user was trying to do: missing template<>.
using A<char> = char; // expected-error {{partial specialization of alias templates is not permitted}}
