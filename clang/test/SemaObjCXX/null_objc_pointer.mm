// RUN: %clang_cc1 -fsyntax-only -verify %s
#define NULL __null

@interface X
@end

void f() {
  bool b;
  X *d;
  b = d < NULL || NULL < d || d > NULL || NULL > d;
  b = d <= NULL || NULL <= d || d >= NULL || NULL >= d;
  b = d == NULL || NULL == d || d != NULL || NULL != d;
}
