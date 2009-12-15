// RUN: %clang_cc1 %s -fsyntax-only -pedantic

void foo() { 
  return foo();
}
