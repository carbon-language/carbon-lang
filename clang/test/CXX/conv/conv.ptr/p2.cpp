// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace pr7801 {
  extern void* x[];
  void* dummy[] = { &x };
}
