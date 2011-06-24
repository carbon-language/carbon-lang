// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-nonfragile-abi -fsyntax-only -fobjc-arc -verify %s
// rdar://9340606

@interface Foo {
@public
    id __unsafe_unretained x;
    id __weak y;
    id __autoreleasing z; // expected-error {{ivars cannot have __autoreleasing ownership}}
}
@property(strong) id x; // expected-note {{property declared here}}
@property(strong) id y; // expected-note {{property declared here}}
@property(strong) id z;
@end

@implementation Foo
@synthesize x; // expected-error {{existing ivar 'x' for strong property 'x' may not be __unsafe_unretained}}
@synthesize y; // expected-error {{existing ivar 'y' for strong property 'y' may not be __weak}}
@synthesize z; // suppressed
@end

@interface Bar {
@public
    id __unsafe_unretained x;
    id __weak y;
    id __autoreleasing z; // expected-error {{ivars cannot have __autoreleasing ownership}}
}
@property(retain) id x; // expected-note {{property declared here}}
@property(retain) id y; // expected-note {{property declared here}}
@property(retain) id z;
@end

@implementation Bar
@synthesize x; // expected-error {{existing ivar 'x' for strong property 'x' may not be __unsafe_unretained}}
@synthesize y; // expected-error {{existing ivar 'y' for strong property 'y' may not be __weak}}
@synthesize z; // suppressed
@end

@interface Bas {
@public
    id __unsafe_unretained x;
    id __weak y;
    id __autoreleasing z; // expected-error {{ivars cannot have __autoreleasing ownership}}
}
@property(copy) id x; // expected-note {{property declared here}}
@property(copy) id y; // expected-note {{property declared here}} 
@property(copy) id z;
@end

@implementation Bas
@synthesize x; // expected-error {{existing ivar 'x' for strong property 'x' may not be __unsafe_unretained}}
@synthesize y; // expected-error {{existing ivar 'y' for strong property 'y' may not be __weak}}
@synthesize z; // suppressed
@end

@interface Bat 
@property(strong) __unsafe_unretained id x; // expected-error {{strong property 'x' may not also be declared __unsafe_unretained}}
@property(strong) __autoreleasing id z; // expected-error {{strong property 'z' may not also be declared __autoreleasing}}
@end

@interface Bau 
@property(retain) __unsafe_unretained id x; // expected-error {{strong property 'x' may not also be declared __unsafe_unretained}}
@property(retain) __autoreleasing id z; // expected-error {{strong property 'z' may not also be declared __autoreleasing}}
@end

@interface Bav 
@property(copy) __unsafe_unretained id x; // expected-error {{strong property 'x' may not also be declared __unsafe_unretained}}
@property(copy) __autoreleasing id z; // expected-error {{strong property 'z' may not also be declared __autoreleasing}}
@end

// rdar://9341593
@interface Gorf  {
   id __unsafe_unretained x;
   id y;
}
@property(assign) id __unsafe_unretained x;
@property(assign) id y; // expected-note {{property declared here}}
@property(assign) id z;
@end

@implementation Gorf
@synthesize x;
@synthesize y; // expected-error {{existing ivar 'y' for unsafe_unretained property 'y' must be __unsafe_unretained}}
@synthesize z;
@end

@interface Gorf2  {
   id __unsafe_unretained x;
   id y;
}
@property(unsafe_unretained) id __unsafe_unretained x;
@property(unsafe_unretained) id y; // expected-note {{property declared here}}
@property(unsafe_unretained) id z;
@end

@implementation Gorf2
@synthesize x;
@synthesize y; // expected-error {{existing ivar 'y' for unsafe_unretained property 'y' must be __unsafe_unretained}}
@synthesize z;
@end

// rdar://9355230
@interface I {
  char _isAutosaving;
}
@property char isAutosaving;

@end

@implementation I
@synthesize isAutosaving = _isAutosaving;
@end

