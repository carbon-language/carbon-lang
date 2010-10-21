// RUN: %clang_cc1  -fsyntax-only -verify %s

@interface Foo 
@property (readonly) char foo; // expected-note {{property declared here}}
@property (readwrite) char bar; // expected-note {{property declared here}}
@end

@interface Foo ()
@property (readwrite) char foo; // OK 
@property (readwrite) char NewProperty; // expected-note 2 {{property declared here}} 
@property (readwrite) char bar; // expected-error{{illegal redeclaration of 'readwrite' property in continuation class 'Foo' (perhaps you intended this to be a 'readwrite' redeclaration of a 'readonly' public property?)}}
@end

@interface Foo ()
@property (readwrite) char foo;	//  OK again, make primary property readwrite for 2nd time!
@property (readwrite) char NewProperty; // expected-error {{redeclaration of property in continuation class 'Foo' (attribute must be 'readwrite', while its primary must be 'readonly')}}
@end

@interface Foo ()
@property (readonly) char foo; // expected-error {{redeclaration of property in continuation class 'Foo' (attribute must be 'readwrite', while its primary must be 'readonly')}}
@property (readwrite) char NewProperty; // expected-error {{redeclaration of property in continuation class 'Foo' (attribute must be 'readwrite', while its primary must be 'readonly')}}
@end

