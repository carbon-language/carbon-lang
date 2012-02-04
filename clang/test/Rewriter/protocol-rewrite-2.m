// RUN: %clang_cc1 -rewrite-objc -fobjc-fragile-abi  %s -o %t.cpp
// RUN: %clang_cc1 -fsyntax-only %t.cpp

// rdar://10234024
@protocol Foo;
@protocol Foo
@end
