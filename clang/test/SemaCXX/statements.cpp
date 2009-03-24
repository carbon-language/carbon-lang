// RUN: clang-cc %s -fsyntax-only -pedantic

void foo() { 
  return foo();
}
