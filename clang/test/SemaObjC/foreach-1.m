// RUN: clang -fsyntax-only -verify %s

@class NSArray;

void f(NSArray *a)
{
    for (int i in a); // expected-error{{selector element type ('int') is not a valid object}}
    for ((id)2 in a); // expected-error{{selector element is not a valid lvalue}}
    for (2 in a); // expected-error{{selector element is not a valid lvalue}}
}