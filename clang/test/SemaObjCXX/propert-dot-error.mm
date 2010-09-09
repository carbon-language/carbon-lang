// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar: // 8379892

struct X {
  X();
  X(const X&);
  ~X();
};

@interface A {
  X xval;
}

- (X)x;
- (void)setx:(X)x;
@end

void f(A* a) {
  a.x = X(); // expected-error {{setter method is needed to assign to object using property assignment syntax}}
}

