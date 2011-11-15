// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct X { 
  void f() const;
  ~X();
};

@interface A {
  X x_;
}

- (const X&)x;
- (void)setx:(const X&)other;
@end

@implementation A

- (const X&)x { return x_; }
- (void)setx:(const X&)other { x_ = other; }
- (void)method {
  self.x.f();
}
@end

// rdar://problem/10444030
@interface Test2
- (void) setY: (int) y;
- (int) z;
@end
void test2(Test2 *a) {
  auto y = a.y; // expected-error {{expected getter method not found on object of type 'Test2 *'}} expected-error {{variable 'y' with type 'auto' has incompatible initializer of type}}
  auto z = a.z;
}
