// RUN: clang -fsyntax-only -verify %s

@interface Foo // expected-error {{previous definition is here}}
@end

float Foo;	// expected-error {{redefinition of 'Foo' as different kind of symbol}}

@class Bar;  // expected-error {{previous definition is here}}

typedef int Bar;  // expected-error {{redefinition of 'Bar' as different kind of symbol}}

@implementation FooBar // expected-warning {{cannot find interface declaration for 'FooBar'}} 
@end


typedef int OBJECT; // expected-error {{previous definition is here}}

@class OBJECT ;	// expected-error {{redefinition of 'OBJECT' as different kind of symbol}}


typedef int Gorf;  // expected-error {{previous definition is here}}

@interface Gorf @end // expected-error {{redefinition of 'Gorf' as different kind of symbol}}

void Gorf() // expected-error {{redefinition of 'Gorf' as different kind of symbol}}
{
	int Bar, Foo, FooBar;
}
