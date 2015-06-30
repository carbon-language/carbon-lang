
// Support parsing of function concepts and variable concepts

// RUN:  %clang_cc1 -std=c++14 -fconcepts-ts -x c++ -verify %s

template<typename T> concept bool C1 = true;

template<typename T> concept bool C2() { return true; }

template<typename T>
struct A { typedef bool Boolean; };

template<int N>
A<void>::Boolean concept C3(!0);

template<typename T, int = 0>
concept auto C4(void) -> bool { return true; }

constexpr int One = 1;

template <typename>
static concept decltype(!0) C5 { bool(One) };

template<typename T> concept concept bool C6 = true; // expected-warning {{duplicate 'concept' declaration specifier}}

template<typename T> concept concept bool C7() { return true; } // expected-warning {{duplicate 'concept' declaration specifier}}

concept D1 = true; // expected-error {{C++ requires a type specifier for all declarations}}

template<concept T> concept bool D2 = true; // expected-error {{unknown type name 'T'}}
