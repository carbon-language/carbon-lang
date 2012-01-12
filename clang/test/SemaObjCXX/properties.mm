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

// rdar://problem/10672108
@interface Test3
- (int) length;
@end
void test3(Test3 *t) {
  char vla[t.length] = {}; // expected-error {{variable-sized object may not be initialized}}
  char *heaparray = new char[t.length];
}

// <rdar://problem/10672501>
namespace std {
  template<typename T> void count();
}

@interface Test4 {
@public
  int count;
}
@property int count;
@end

void test4(Test4* t4) {
  if (t4.count < 2) { }
  if (t4->count < 2) { }
}

