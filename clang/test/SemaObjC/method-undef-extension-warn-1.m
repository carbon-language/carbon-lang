// RUN: clang-cc -fsyntax-only -verify %s

@interface MyClass
@end

@protocol P
- (void)Pmeth;
- (void)Pmeth1;
@end

// Class extension
@interface MyClass () <P>
- (void)meth2;
@end

// Add a category to test that clang does not emit warning for this method.
@interface MyClass (Category) 
- (void)categoryMethod;
@end

@implementation MyClass // expected-warning {{incomplete implementation}} \
                           expected-warning {{method definition for 'meth2' not found}} \
                           expected-warning {{method definition for 'Pmeth1' not found}}
- (void)Pmeth {}
@end
