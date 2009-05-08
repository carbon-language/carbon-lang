// RUN: clang-cc  -fsyntax-only -verify %s

@interface I
+ new; // expected-note {{method 'new' is used for the forward class}}
@end
Class isa;

@class NotKnown;

void foo(NotKnown *n) {
  [isa new];
  [NotKnown new];	   /* expected-warning {{receiver 'NotKnown' is a forward class and corresponding}} */
}
