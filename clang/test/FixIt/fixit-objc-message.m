// Objective-C recovery
// RUN: cp %s %t
// RUN: %clang_cc1 -pedantic -Wall -fixit -x objective-c %t || true
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wall -Werror -x objective-c %t

// Objective-C++ recovery
// RUN: cp %s %t
// RUN: %clang_cc1 -pedantic -Wall -fixit -x objective-c++ %t || true
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wall -Werror -x objective-c++ %t

@interface A
- (int)method1:(int)x second:(float)y;
+ (int)method2:(int)x second:(double)y;
- (int)getBlah;
@end

void f(A *a, int i, int j) {
  a method1:5+2 second:+(3.14159)];
  a method1:[a method1:3 second:j] second:i++]
  a getBlah];
}
