// RUN: clang -fsyntax-only -verify %s

@protocol SomeProtocol
@end

void foo(id x) {
  bar((short<SomeProtocol>)x); // expected-error {{expected ')'}} expected-error {{to match this '('}}
  bar((<SomeProtocol>)x);      // expected-warning {{protocol qualifiers without 'id' is archaic}}
}

