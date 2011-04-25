// Don't crash.

// RUN: %clang_cc1 -DCAKE -x c-header %S/modified-header-crash.h -emit-pch -o %t
// RUN: touch %S/modified-header-crash.h
// RUN: not %clang_cc1 %s -include-pch %t -fsyntax-only

void f(void) {
  foo = 3;
}
