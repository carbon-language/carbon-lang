// RUN: %clang_cc1 -analyze -std=c++11 -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-config cfg-temporary-dtors=true -Wno-null-dereference -verify %s
// expected-no-diagnostics

#include "Inputs/system-header-simulator-cxx.h"

namespace Cxx11BraceInit {
  struct Foo {
    ~Foo() {}
  };

  void testInitializerList() {
    for (Foo foo : {Foo(), Foo()}) {}
  }
}

