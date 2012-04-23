// RUN: %clang_cc1 -triple x86_64-apple-darwin9.0.0 -fsyntax-only -verify %s

@protocol P
- (void)proto_method __attribute__((availability(macosx,introduced=10.1,deprecated=10.2))); // expected-note 2 {{method 'proto_method' declared here}}
@end

@interface A <P>
- (void)method __attribute__((availability(macosx,introduced=10.1,deprecated=10.2))); // expected-note {{method 'method' declared here}}
@end

@interface B : A
- (void)method;
@end

void f(A *a, B *b) {
  [a method]; // expected-warning{{'method' is deprecated: first deprecated in Mac OS X 10.2}}
  [b method];
  [a proto_method]; // expected-warning{{'proto_method' is deprecated: first deprecated in Mac OS X 10.2}}
  [b proto_method]; // expected-warning{{'proto_method' is deprecated: first deprecated in Mac OS X 10.2}}
}
