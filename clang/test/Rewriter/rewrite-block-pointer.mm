// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: FileCheck -check-prefix LP --input-file=%t-rw.cpp %s
// radar 7638400

@interface X
@end

void foo(void (^block)(int));

@implementation X
static void enumerateIt(void (^block)(id, id, char *)) {
      foo(^(int idx) { });
}
@end

// CHECK-LP: static void enumerateIt(void (*)(id, id, char *));
