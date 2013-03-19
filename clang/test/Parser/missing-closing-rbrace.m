// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar: //6854840
@interface A {@end // expected-error {{'@end' appears where closing brace '}' is expected}}\
                   // expected-note {{to match this '{'}}\
                   // expected-note {{class started here}}
		   // expected-error {{expected '}'}} expected-error {{missing '@end'}}
