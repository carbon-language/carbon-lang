// RUN: clang-cc %s -fsyntax-only -verify -fms-extensions

void f() throw(...) { }
