// RUN: clang -fblocks -fnext-runtime --emit-llvm -o %t %s -verify

@class Foo;
@protocol P;

void t1()
{
	__block int a;
	^{ a = 10; }(); // expected-error {{cannot compile this __block variable in block literal yet}}
	
	void (^block)(void);
    ^{ (void)block; }(); // expected-error {{}}

    struct Foo *__attribute__ ((NSObject)) foo;
    ^{ (void)foo; }(); // expected-error {{cannot compile this __attribute__((NSObject)) variable in block literal yet}}
    
    typedef struct CGColor * __attribute__ ((NSObject)) CGColorRef;
    CGColorRef color;
    ^{ (void)color; }(); // expected-error {{cannot compile this __attribute__((NSObject)) variable in block literal yet}}
    
    id a1;
    ^{ (void)a1; }(); // expected-error {{cannot compile this Objective-C variable in block literal yet}}
    
    Foo *a2;
    ^{ (void)a2; }(); // expected-error {{cannot compile this Objective-C variable in block literal yet}}
    
    id<P> a3;
    ^{ (void)a3; }(); // expected-error {{cannot compile this Objective-C variable in block literal yet}}

    Foo<P> *a4;
    ^{ (void)a4; }(); // expected-error {{cannot compile this Objective-C variable in block literal yet}}
    
    
}
