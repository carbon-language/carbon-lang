// RUN: clang %s -verify -fsyntax-only

@interface NSConstantString;
@end



NSConstantString *s = @"123"; // simple
NSConstantString *t = @"123" @"456"; // concat
NSConstantString *u = @"123" @ blah; // expected-error: {{unexpected token}}

