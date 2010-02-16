// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: FileCheck -check-prefix LP --input-file=%t-rw.cpp %s
// radar 7638400

// FIXME. Arrange this test's rewritten source to compile with clang
@interface X
@end

void foo(void (^block)(int));

@implementation X
static void enumerateIt(void (^block)(id, id, char *)) {
      foo(^(int idx) { });
}
@end

// CHECK-LP: static void enumerateIt(void (*)(id, id, char *));

// radar 7651312
void apply(void (^block)(int));

static void x(int (^cmp)(int, int)) {
	x(cmp);
}

static void y(int (^cmp)(int, int)) {
	apply(^(int sect) {
		x(cmp);
    });
}
