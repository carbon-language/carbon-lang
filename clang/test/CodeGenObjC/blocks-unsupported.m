// RUN: clang -fnext-runtime --emit-llvm -o %t %s -verify

@class Foo;
@protocol P;

void t1()
{
	__block int a;
	^{ a = 10; }(); // expected-error {{cannot compile this block expression with __block variables yet}}
	
	void (^block)(void);
    ^{ (void)block; }(); // expected-error {{}}

    struct Foo *__attribute__ ((NSObject)) foo;
    ^{ (void)foo; }(); // expected-error {{cannot compile this block expression with __attribute__((NSObject)) variable yet}}
    
    typedef struct CGColor * __attribute__ ((NSObject)) CGColorRef;
    CGColorRef color;
    ^{ (void)color; }(); // expected-error {{cannot compile this block expression with __attribute__((NSObject)) variable yet}}
    
    id a1;
    ^{ (void)a1; }(); // expected-error {{cannot compile this block expression with Objective-C variable yet}}
    
    Foo *a2;
    ^{ (void)a2; }(); // expected-error {{cannot compile this block expression with Objective-C variable yet}}
    
    id<P> a3;
    ^{ (void)a3; }(); // expected-error {{cannot compile this block expression with Objective-C variable yet}}

    Foo<P> *a4;
    ^{ (void)a4; }(); // expected-error {{cannot compile this block expression with Objective-C variable yet}}
    
    
}
