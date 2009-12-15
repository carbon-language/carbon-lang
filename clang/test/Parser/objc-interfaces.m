// RUN: %clang_cc1 %s -fsyntax-only -verify

// Test features and error recovery for objc interfaces.

@interface INTF
- (int*) foo2  __attribute__((deprecated)) : (int) x1 __attribute__((deprecated)); // expected-error {{expected ';' after method prototype}}
@end

