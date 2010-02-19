// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface I1
- (void)method;
@end

@implementation I1
- (void)method {
  struct x { };
  [x method]; // expected-error{{invalid receiver to message expression}}
}
@end
