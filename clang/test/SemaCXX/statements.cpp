// RUN: clang %s -fsyntax-only -pedantic

void foo() { 
  return foo();
}
