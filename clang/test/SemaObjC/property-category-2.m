// RUN: %clang_cc1 -fsyntax-only -verify %s
// Test that a property can be synthesize in a category
// implementation with no error.

@protocol MyProtocol
@property float  myFloat;
@property float  anotherFloat; // expected-warning {{property 'anotherFloat' requires method 'anotherFloat' to be defined - use @dynamic}} \
                               // expected-warning {{property 'anotherFloat' requires method 'setAnotherFloat:' to be defined }}
@end

@interface MyObject { float anotherFloat; }
@end

@interface MyObject (CAT) <MyProtocol>
@end

@implementation MyObject (CAT)	// expected-note 2 {{implementation is here}}
@dynamic myFloat;	// OK
@synthesize anotherFloat; // expected-error {{@synthesize not allowed in a category's implementation}}
@end
