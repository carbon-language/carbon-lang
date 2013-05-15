// RUN: %clang_cc1 -fsyntax-only -verify -Wsign-conversion %s
// rdar://13855394

@interface NSObject
- new;
@end

@interface X : NSObject
@property unsigned int uint;
@end

@implementation X 
@synthesize uint;
@end

void foo() {
    X *x = [X new];
    signed int sint = -1;
    [x setUint:sint];  // expected-warning {{implicit conversion changes signedness: 'int' to 'unsigned int'}}
    x.uint = sint; // expected-warning {{implicit conversion changes signedness: 'int' to 'unsigned int'}}
}
