// RUN: %clang_cc1 -fsyntax-only -verify -cxx-abi microsoft -Wno-objc-root-class %s

class Foo {
  ~Foo(); // expected-note {{implicitly declared private here}}
};

@interface bar
- (void) my_method: (Foo)arg;
@end

@implementation bar
- (void) my_method: (Foo)arg { // expected-error {{variable of type 'Foo' has private destructor}}
}
@end
