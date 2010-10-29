// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface MyClass // expected-note {{required for direct or indirect protocol 'P'}}
@end

@protocol P
- (void)Pmeth;
- (void)Pmeth1; // expected-note {{method declared here}}
@end

// Class extension
@interface MyClass () <P>
- (void)meth2; // expected-note {{method definition for 'meth2' not found}}
@end

// Add a category to test that clang does not emit warning for this method.
@interface MyClass (Category) 
- (void)categoryMethod;
@end

@implementation MyClass // expected-warning {{incomplete implementation}}  \
			// expected-warning {{method in protocol not implemented [-Wprotocol]}}
- (void)Pmeth {}
@end
