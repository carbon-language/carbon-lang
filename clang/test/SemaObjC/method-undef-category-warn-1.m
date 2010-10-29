// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface MyClass1
@end

@protocol P
- (void) Pmeth;	  // expected-note {{method declared here }}
- (void) Pmeth1;    // expected-note {{method declared here }}
@end

@interface MyClass1(CAT) <P> // expected-note {{required for direct or indirect protocol 'P'}}
- (void) meth2;	 // expected-note {{method definition for 'meth2' not found}}
@end

@implementation MyClass1(CAT) // expected-warning {{incomplete implementation}}  \
				// expected-warning {{method in protocol not implemented [-Wprotocol]}}
- (void) Pmeth1{}
@end

@interface MyClass1(DOG) <P> // expected-note {{required for direct or indirect protocol 'P'}}
- (void)ppp;    // expected-note {{method definition for 'ppp' not found}} 
@end

@implementation MyClass1(DOG) // expected-warning {{incomplete implementation}} \
		// expected-warning {{method in protocol not implemented [-Wprotocol]}}
- (void) Pmeth {}
@end

@implementation MyClass1(CAT1)
@end
