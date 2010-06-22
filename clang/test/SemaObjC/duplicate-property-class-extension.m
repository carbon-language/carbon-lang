// RUN: %clang_cc1  -fsyntax-only -verify %s

@interface Foo 
@property (readonly) char foo; // expected-note {{property declared here}}
@end

@interface Foo ()
@property (readwrite) char foo; // OK 
@property (readwrite) char NewProperty; // expected-note 2 {{property declared here}} 
@end

@interface Foo ()
@property (readwrite) char foo;	//  OK again, make primary property readwrite for 2nd time!
@property (readwrite) char NewProperty; // expected-error {{illegal declaration of property in continuation class 'Foo': attribute must be readwrite, while its primary must be readonly}}
@end

@interface Foo ()
@property (readonly) char foo;	// expected-error {{illegal declaration of property in continuation class 'Foo': attribute must be readwrite, while its primary must be readonly}}
@property (readwrite) char NewProperty; // expected-error {{illegal declaration of property in continuation class 'Foo': attribute must be readwrite, while its primary must be readonly}}
@end

