// RUN: %clang_cc1 %s -fsyntax-only -verify

// "During the lookup for a base class name, non-type names are ignored"
namespace PR5840 {
  struct Base {};
  int Base = 10;
  struct Derived : Base {};
}
