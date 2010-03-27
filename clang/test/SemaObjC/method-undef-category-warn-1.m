// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface MyClass1  // expected-note 2 {{required for direct or indirect protocol 'P'}}
@end

@protocol P
- (void) Pmeth;	 // expected-note {{method definition for 'Pmeth' not found}}
- (void) Pmeth1;   // expected-note {{method definition for 'Pmeth1' not found}}
@end

@interface MyClass1(CAT) <P>
- (void) meth2;	 // expected-note {{method definition for 'meth2' not found}}
@end

@implementation MyClass1(CAT) // expected-warning {{incomplete implementation}} 
- (void) Pmeth1{}
@end

@interface MyClass1(DOG) <P>
- (void)ppp;    // expected-note {{method definition for 'ppp' not found}}
@end

@implementation MyClass1(DOG) // expected-warning {{incomplete implementation}}
- (void) Pmeth {}
@end

@implementation MyClass1(CAT1)
@end
