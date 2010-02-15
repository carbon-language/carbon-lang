// RUN: %clang_cc1  -fsyntax-only -verify %s

@interface Foo 
@property (readonly) char foo;
@end

@interface Foo ()
@property (readwrite) char foo; // expected-note {{property declared here}}
@end

@interface Foo ()
@property (readwrite) char foo;	// expected-error {{property has a previous declaration}}
@end
