// RUN: clang -fsyntax-only -verify %s

@class FOO, BAR;
@class FOO, BAR;

@interface INTF : FOO	// expected-error {{cannot find interface declaration for 'FOO', superclass of 'INTF'}}
@end

@interface FOO 
- (BAR*) Meth1;
- (FOO*) Meth2;
@end

@interface INTF1 : FOO	
@end

@interface INTF2 : INTF1
@end


@class INTF1, INTF2;

@interface INTF2 : INTF1 // expected-error {{duplicate interface declaration for class 'INTF2'}}
@end
