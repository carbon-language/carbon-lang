// RUN: %clang_cc1 -triple x86_64-apple-macosx10.8.0 -fsyntax-only -verify %s

// This test case shows that 'availablity' and 'deprecated' does not inherit
// when a property is redeclared in a subclass.  This is intentional.

@interface NSObject @end
@protocol myProtocol
@property int myProtocolProperty __attribute__((availability(macosx,introduced=10.7,deprecated=10.8)));
@end

@interface Foo : NSObject
@property int myProperty __attribute__((availability(macosx,introduced=10.7,deprecated=10.8)));  // expected-note {{'myProperty' declared here}} \
								// expected-note {{method 'myProperty' declared here}} \
								// expected-note {{property 'myProperty' is declared deprecated here}}
@end

@interface Bar : Foo <myProtocol>
@property int myProperty; // expected-note {{'myProperty' declared here}}
@property int myProtocolProperty; // expected-note {{'myProtocolProperty' declared here}}
@end

void test(Foo *y, Bar *x) {
  y.myProperty = 0; // expected-warning {{'myProperty' is deprecated: first deprecated in OS X 10.8}}
  [y myProperty];   // expected-warning {{'myProperty' is deprecated: first deprecated in OS X 10.8}} 

  x.myProperty = 1; // no-warning
  [x myProperty]; // expected-warning {{'myProperty' is deprecated: first deprecated in OS X 10.8}}

  x.myProtocolProperty = 0; // no-warning

  [x myProtocolProperty]; // expected-warning {{'myProtocolProperty' is deprecated: first deprecated in OS X 10.8}}
}
