// RUN: %clang_cc1 -Wmissing-method-return-type -fsyntax-only -verify %s
// rdar://9615045

@interface I
-  initWithFoo:(id)foo; // expected-warning {{method has no return type specified; defaults to 'id' [-Wmissing-method-return-type]}}
@end

@implementation I
- initWithFoo:(id)foo { return 0; } // expected-warning {{method has no return type specified; defaults to 'id' [-Wmissing-method-return-type]}}
@end

