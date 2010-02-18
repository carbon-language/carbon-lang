// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// radar 7638400

typedef void * id;

@interface X
@end

void foo(void (^block)(int));

@implementation X
static void enumerateIt(void (^block)(id, id, char *)) {
      foo(^(int idx) { });
}
@end

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

// radar 7659483
void *_Block_copy(const void *aBlock);
void x(void (^block)(void)) {
        block = ((__typeof(block))_Block_copy((const void *)(block)));
}
