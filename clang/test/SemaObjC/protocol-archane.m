// RUN: clang -fsyntax-only -verify %s
// rdar://5986251

@protocol SomeProtocol
- (void) bar;
@end

void foo(id x) {
  bar((short<SomeProtocol>)x); // expected-error {{expected ')'}} expected-error {{to match this '('}}
  bar((<SomeProtocol>)x);      // expected-warning {{protocol qualifiers without 'id' is archaic}}

  [(<SomeProtocol>)x bar];      // expected-warning {{protocol qualifiers without 'id' is archaic}}
}

@protocol MyProtocol
- (void)doSomething;
@end

@interface MyClass
- (void)m1:(id <MyProtocol> const)arg1;

// FIXME: provide a better diagnostic (no typedef).
- (void)m2:(id <MyProtocol> short)arg1; // expected-error {{'short typedef' is invalid}}
@end