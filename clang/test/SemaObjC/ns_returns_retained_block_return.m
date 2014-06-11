// RUN: %clang_cc1  -fblocks -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fblocks -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1  -fblocks -fobjc-arc -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fblocks -fobjc-arc -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics
// rdar://17259812

typedef void (^BT) ();

BT foo()  __attribute__((ns_returns_retained));

@interface I
BT foo()  __attribute__((ns_returns_retained));
@end

@implementation I
BT foo()  __attribute__((ns_returns_retained)) {return ^{}; }
@end
