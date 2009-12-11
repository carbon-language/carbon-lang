// RUN: clang-cc -fsyntax-only -verify -fblocks %s
@protocol NSObject;

void bar(id(^)(void));
void foo(id <NSObject>(^objectCreationBlock)(void)) {
    return bar(objectCreationBlock); // expected-warning{{incompatible pointer types passing 'id (^)()', expected 'id<NSObject> (^)()'}}
}

void bar2(id(*)(void));
void foo2(id <NSObject>(*objectCreationBlock)(void)) {
    return bar2(objectCreationBlock); // expected-warning{{incompatible pointer types passing 'id (*)()', expected 'id<NSObject> (*)()'}}
}

void bar3(id(*)()); // expected-note{{candidate function}}
void foo3(id (*objectCreationBlock)(int)) {
    return bar3(objectCreationBlock); // expected-error{{no matching}}
}

void bar4(id(^)()); // expected-note{{candidate function}}
void foo4(id (^objectCreationBlock)(int)) {
    return bar4(objectCreationBlock); // expected-error{{no matching}}
}

void foo5(id (^x)(int)) {
  if (x) { }
}

// <rdar://problem/6590445>
@interface Foo {
    @private
    void (^_block)(void);
}
- (void)bar;
@end

namespace N {
  class X { };      
  void foo(X);
}

@implementation Foo
- (void)bar {
    _block();
    foo(N::X()); // okay
}
@end

typedef signed char BOOL;
void foo6(void *block) {  
	void (^vb)(id obj, int idx, BOOL *stop) = (void (^)(id, int, BOOL *))block;
    BOOL (^bb)(id obj, int idx, BOOL *stop) = (BOOL (^)(id, int, BOOL *))block;
}
