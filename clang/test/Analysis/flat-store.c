// RUN: %clang_cc1 -analyze -analyzer-check-objc-mem -analyzer-store=flat -verify %s
#define FAIL ((void)*(char*)0)
struct simple { int x; };

void PR7297 () {
  struct simple a;
  struct simple *p = &a;
  p->x = 5;
  if (!p[0].x) FAIL; // no-warning
  if (p[0].x) FAIL; // expected-warning {{null}}
}
