// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o %t.cpp
// RUN: %clang_cc1 -fsyntax-only %t.cpp

// rdar://10234024
@protocol Foo;
@protocol Foo
@end
