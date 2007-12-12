// RUN: clang %s -verify -fsyntax-only

@class NSString;
@interface NSConstantString;
@end



NSString *s = @"123"; // simple
NSString *t = @"123" @"456"; // concat
NSString *u = @"123" @ blah; // expected-error: {{unexpected token}}

