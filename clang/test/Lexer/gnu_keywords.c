// RUN: %clang -DGNU_KEYWORDS -std=gnu89 -fsyntax-only -verify %s
// RUN: %clang -DGNU_KEYWORDS -std=c99 -fgnu-keywords -fsyntax-only -verify %s
// RUN: %clang -std=c99 -fsyntax-only -verify %s
// RUN: %clang -std=gnu89 -fno-gnu-keywords -fsyntax-only -verify %s

void f() {
#ifdef GNU_KEYWORDS
  asm ("ret" : :);
#else
  int asm;
#endif
}
