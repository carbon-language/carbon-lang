// RUN: %clang_cc1 -fsyntax-only -std=c++1y -DCXX1Y -Wc++98-compat-pedantic -verify %s -DCXX1Y2
// RUN: %clang_cc1 -fsyntax-only -std=c++1y -DCXX1Y -Wc++98-compat -Werror %s -DCXX1Y2
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wc++98-compat-pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wc++98-compat -Werror %s
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -Werror %s

// RUN: %clang_cc1 -fsyntax-only -std=c++1y -Wc++98-compat-pedantic -verify %s -Wno-c++98-c++11-compat-pedantic -DCXX1Y2

// -Wc++98-compat-pedantic warns on C++11 features which we accept without a
// warning in C++98 mode.

#line 32767 // ok
#line 32768 // expected-warning {{#line number greater than 32767 is incompatible with C++98}}

#define VA_MACRO(x, ...) x // expected-warning {{variadic macros are incompatible with C++98}}
VA_MACRO(,x) // expected-warning {{empty macro arguments are incompatible with C++98}}

; // expected-warning {{extra ';' outside of a function is incompatible with C++98}}

enum Enum {
  Enum_value, // expected-warning {{commas at the end of enumerator lists are incompatible with C++98}}
};

template<typename T> struct InstantiationAfterSpecialization {};
template<> struct InstantiationAfterSpecialization<int> {}; // expected-note {{here}}
template struct InstantiationAfterSpecialization<int>; // expected-warning {{explicit instantiation of 'InstantiationAfterSpecialization<int>' that occurs after an explicit specialization is incompatible with C++98}}

void *dlsym();
void (*FnPtr)() = (void(*)())dlsym(); // expected-warning {{cast between pointer-to-function and pointer-to-object is incompatible with C++98}}
void *FnVoidPtr = (void*)&dlsym; // expected-warning {{cast between pointer-to-function and pointer-to-object is incompatible with C++98}}

struct ConvertToInt {
  operator int();
};
int *ArraySizeConversion = new int[ConvertToInt()];
#ifdef CXX1Y2
// expected-warning@-2 {{implicit conversion from array size expression of type 'ConvertToInt' to integral type 'size_t' is incompatible with C++98}}
#else
// expected-warning@-4 {{implicit conversion from array size expression of type 'ConvertToInt' to integral type 'int' is incompatible with C++98}}
#endif

template<typename T> class ExternTemplate {};
extern template class ExternTemplate<int>; // expected-warning {{extern templates are incompatible with C++98}}

long long ll1 = // expected-warning {{'long long' is incompatible with C++98}}
         -42LL; // expected-warning {{'long long' is incompatible with C++98}}
unsigned long long ull1 = // expected-warning {{'long long' is incompatible with C++98}}
                   42ULL; // expected-warning {{'long long' is incompatible with C++98}}

int k = 0b1001;
#ifdef CXX1Y
// expected-warning@-2 {{binary integer literals are incompatible with C++ standards before C++1y}}
#endif

void f(int n) { int a[n]; }
#ifdef CXX1Y
// expected-warning@-2 {{arrays of runtime bound are incompatible with C++ standards before C++1y}}
#endif
