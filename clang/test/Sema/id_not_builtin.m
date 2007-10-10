// RUN: clang %s -parse-ast -verify

id obj; // expected-error{{expected '=', ',', ';', 'asm', or '__attribute__' after declarator}}

@interface Foo

- defaultToId; // expected-error{{cannot find definition of 'id'}}

@end
