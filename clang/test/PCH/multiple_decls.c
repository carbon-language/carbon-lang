// Test this without pch.
// RUN: clang-cc -include %S/multiple_decls.h -fsyntax-only -ast-print -o - %s &&

// Test with pch.
// RUN: clang-cc -emit-pch -o %t %S/multiple_decls.h &&
// RUN: clang-cc -include-pch %t -fsyntax-only -ast-print -o - %s 

void f0(char c) {
  wide(c);
}

struct wide w;
struct narrow n;

void f1(int i) {
  narrow(i);
}
