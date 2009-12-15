// RUN: %clang_cc1 -fsyntax-only -verify %s -triple i686-pc-linux-gnu

int printf(char const*,...);
void percentm(void) {
  printf("%m");
}
