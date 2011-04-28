// Don't crash.

// RUN: cp %S/modified-header-crash.h %t.h
// RUN: %clang_cc1 -DCAKE -x c-header %t.h -emit-pch -o %t
// RUN: echo >> %t.h
// RUN: not %clang_cc1 %s -include-pch %t -fsyntax-only

// FIXME: On Windows we don't detect that the header was modified ?
// XFAIL: win32

void f(void) {
  foo = 3;
}
