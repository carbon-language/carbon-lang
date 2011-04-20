// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar: // 8379892

struct X {
  X();
  X(const X&);
  ~X();

  static int staticData;
  int data;
  void method();
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

struct Y : X { };

@interface B {
@private
  Y *y;
}
- (Y)value;
- (void)setValue : (Y) arg;
@property Y value;
@end

void g(B *b) {
  b.value.data = 17; // expected-error {{not assignable}}
  b.value.staticData = 17;
  b.value.method();
}

@interface C
@end

@implementation C
- (void)method:(B *)b {
  // <rdar://problem/8985943>
  b.operator+ = 17; // expected-error{{'operator+' is not a valid property name (accessing an object of type 'B *')}}
}
@end
