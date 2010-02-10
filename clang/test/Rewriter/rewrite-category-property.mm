// RUN: %clang_cc1 -x objective-c++ -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: FileCheck -check-prefix LP --input-file=%t-rw.cpp %s
// radar 7630636

@class Y, Z;

@interface A
@property (readonly) Y *y;
@end

@interface A (cat)
@property (readonly) Z *z;
@end

// CHECK-LP: // @property (readonly) Z *z;
