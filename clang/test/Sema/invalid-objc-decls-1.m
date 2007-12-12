// RUN: clang -fsyntax-only -verify %s

@interface Super @end
Super s1; // expected-error{{statically allocated Objective-C object 's1'}}

extern Super e1; // expected-error{{statically allocated Objective-C object 'e1'}}

struct S {
  Super s1; // expected-error{{statically allocated Objective-C object 's1'}}
};

@protocol P1 @end

@interface INTF
{
  Super ivar1; // expected-error{{statically allocated Objective-C object 'ivar1'}}
}
@end

@interface MyIntf
{
  Super<P1> ivar1; // expected-error{{statically allocated Objective-C object 'ivar1'}}
}
@end

Super foo(Super parm1) {
	Super p1; // expected-error{{statically allocated Objective-C object 'p1'}}
	return p1;
}
