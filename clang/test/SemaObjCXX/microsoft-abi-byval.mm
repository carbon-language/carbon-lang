// RUN: %clang_cc1 -fsyntax-only -verify -cxx-abi microsoft -Wno-objc-root-class %s
// expected-no-diagnostics

class Foo {
  ~Foo();
};

@interface bar
- (void) my_method: (Foo)arg;
@end

@implementation bar
- (void) my_method: (Foo)arg { // no error; MS ABI will call Foo's dtor, but we skip the access check.
}
@end
