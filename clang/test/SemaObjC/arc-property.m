// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-nonfragile-abi -fobjc-runtime-has-weak -fsyntax-only -fobjc-arc -fblocks -verify %s
// rdar://9309489

@interface MyClass {
        id __weak myString;
        id StrongIvar;
        id __weak myString2;
        id __weak myString3;
        id StrongIvar5;
}
@property (strong) id myString; // expected-note {{property declared here}}
@property (strong) id myString1;
@property (retain) id myString2; // expected-note {{property declared here}}
//
@property (weak) id myString3;
@property (weak) id myString4;
@property __weak id myString5; // expected-note {{property declared here}}
@end

@implementation MyClass
@synthesize myString; // expected-error {{existing ivar 'myString' for strong property 'myString' may not be __weak}}
@synthesize myString1 = StrongIvar; // OK
@synthesize myString2 = myString2; // expected-error {{existing ivar 'myString2' for strong property 'myString2' may not be __weak}}
//
@synthesize myString3; // OK
@synthesize myString4; // OK
@synthesize myString5 = StrongIvar5; // expected-error {{existing ivar 'StrongIvar5' for __weak property 'myString5' must be __weak}}

@end

// rdar://9340692
@interface Foo {
@public
    id __unsafe_unretained x;   // should be __weak
    id __strong y;
    id __autoreleasing z; // expected-error {{ivars cannot have __autoreleasing ownership}}
}
@property(weak) id x; // expected-note {{property declared here}}
@property(weak) id y; // expected-note {{property declared here}}
@property(weak) id z;
@end

@implementation Foo
@synthesize x;	// expected-error {{existing ivar 'x' for __weak property 'x' must be __weak}}
@synthesize y;	// expected-error {{existing ivar 'y' for __weak property 'y' must be __weak}}
@synthesize z;  // suppressed
@end

