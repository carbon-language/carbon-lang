// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

void bar();
namespace foo { using ::bar; }
using foo::bar;
void bar() {}

void f();
using ::f;
void f() {}
