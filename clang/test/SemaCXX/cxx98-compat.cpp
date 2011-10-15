// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wc++98-compat -verify %s

template<typename ...T>  // expected-warning {{variadic templates are incompatible with C++98}}
class Variadic1 {};

template<template<typename> class ...T>  // expected-warning {{variadic templates are incompatible with C++98}}
class Variadic2 {};

template<int ...I>  // expected-warning {{variadic templates are incompatible with C++98}}
class Variadic3 {};

int alignas(8) with_alignas; // expected-warning {{'alignas' is incompatible with C++98}}
int with_attribute [[ ]]; // expected-warning {{attributes are incompatible with C++98}}

void Literals() {
  (void)u8"str"; // expected-warning {{unicode literals are incompatible with C++98}}
  (void)u"str"; // expected-warning {{unicode literals are incompatible with C++98}}
  (void)U"str"; // expected-warning {{unicode literals are incompatible with C++98}}
  (void)u'x'; // expected-warning {{unicode literals are incompatible with C++98}}
  (void)U'x'; // expected-warning {{unicode literals are incompatible with C++98}}

  (void)u8R"X(str)X"; // expected-warning {{raw string literals are incompatible with C++98}}
  (void)uR"X(str)X"; // expected-warning {{raw string literals are incompatible with C++98}}
  (void)UR"X(str)X"; // expected-warning {{raw string literals are incompatible with C++98}}
  (void)R"X(str)X"; // expected-warning {{raw string literals are incompatible with C++98}}
  (void)LR"X(str)X"; // expected-warning {{raw string literals are incompatible with C++98}}
}

template<typename T> struct S {};
namespace TemplateParsing {
  S<::S<void> > s; // expected-warning {{'<::' is treated as digraph '<:' (aka '[') followed by ':' in C++98}}
  S< ::S<void>> t; // expected-warning {{consecutive right angle brackets are incompatible with C++98 (use '> >')}}
}

void Lambda() {
  []{}; // expected-warning {{lambda expressions are incompatible with C++98}}
}

int InitList() {
  (void)new int {}; // expected-warning {{generalized initializer lists are incompatible with C++98}}
  (void)int{}; // expected-warning {{generalized initializer lists are incompatible with C++98}}
  int x {}; // expected-warning {{generalized initializer lists are incompatible with C++98}}
  return {}; // expected-warning {{generalized initializer lists are incompatible with C++98}}
}

int operator""_hello(const char *); // expected-warning {{literal operators are incompatible with C++98}}

enum EnumFixed : int { // expected-warning {{enumeration types with a fixed underlying type are incompatible with C++98}}
};

enum class EnumScoped { // expected-warning {{scoped enumerations are incompatible with C++98}}
};

void Deleted() = delete; // expected-warning {{deleted function definitions are incompatible with C++98}}
struct Defaulted {
  Defaulted() = default; // expected-warning {{defaulted function definitions are incompatible with C++98}}
};

int &&RvalueReference = 0; // expected-warning {{rvalue references are incompatible with C++98}}
struct RefQualifier {
  void f() &; // expected-warning {{reference qualifiers on functions are incompatible with C++98}}
};

auto f() -> int; // expected-warning {{trailing return types are incompatible with C++98}}

void RangeFor() {
  int xs[] = {1, 2, 3};
  for (int &a : xs) { // expected-warning {{range-based for loop is incompatible with C++98}}
  }
}

struct InClassInit {
  int n = 0; // expected-warning {{in-class initialization of non-static data members is incompatible with C++98}}
};

struct OverrideControlBase {
  virtual void f();
  virtual void g();
};
struct OverrideControl final : OverrideControlBase { // expected-warning {{'final' keyword is incompatible with C++98}}
  virtual void f() override; // expected-warning {{'override' keyword is incompatible with C++98}}
  virtual void g() final; // expected-warning {{'final' keyword is incompatible with C++98}}
};

using AliasDecl = int; // expected-warning {{alias declarations are incompatible with C++98}}
template<typename T> using AliasTemplate = T; // expected-warning {{alias declarations are incompatible with C++98}}

inline namespace N { // expected-warning {{inline namespaces are incompatible with C++98}}
}
