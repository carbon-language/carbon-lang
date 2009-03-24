// RUN: clang-cc -fsyntax-only -verify %s

@interface MyClass1 
@end

@protocol P
- (void) Pmeth;	
- (void) Pmeth1;  
@end

@interface MyClass1(CAT) <P>
- (void) meth2;	
@end

@implementation MyClass1(CAT) // expected-warning {{incomplete implementation}} \
                                 expected-warning {{method definition for 'meth2' not found}} \
                                 expected-warning {{method definition for 'Pmeth' not found}}
- (void) Pmeth1{}
@end

@interface MyClass1(DOG) <P>
- (void)ppp;   
@end

@implementation MyClass1(DOG) // expected-warning {{incomplete implementation}} \
                                 expected-warning {{method definition for 'ppp' not found}} \
                                 expected-warning {{method definition for 'Pmeth1' not found}}
- (void) Pmeth {}
@end

@implementation MyClass1(CAT1)
@end
