// Test this without pch.
// RUN: clang-cc -include %S/builtins.h -fsyntax-only -verify %s

// Test with pch.
// RUN: clang-cc -emit-pch -o %t %S/builtins.h
// RUN: clang-cc -include-pch %t -fsyntax-only -verify %s 

void hello() {
  printf("Hello, World!");
}
