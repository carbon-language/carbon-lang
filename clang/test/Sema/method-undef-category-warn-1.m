// RUN: clang -fsyntax-only -verify %s

@interface MyClass1 
@end

@protocol P
- (void) Pmeth;	// expected-warning {{method definition for 'Pmeth' not found}}
- (void) Pmeth1; // expected-warning {{method definition for 'Pmeth1' not found}}
@end

@interface MyClass1(CAT) <P>
- (void) meth2;	// expected-warning {{method definition for 'meth2' not found}}
@end

@implementation MyClass1(CAT) // expected-warning {{incomplete implementation of category 'CAT'}}
- (void) Pmeth1{}
@end

@interface MyClass1(DOG) <P>
- (void)ppp;  // expected-warning {{method definition for 'ppp' not found}}
@end

@implementation MyClass1(DOG) // expected-warning {{incomplete implementation of category 'DOG'}}
- (void) Pmeth {}
@end

@implementation MyClass1(CAT1)
@end
