// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-extensions

void f() throw(...) { }
