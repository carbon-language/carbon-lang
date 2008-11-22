// RUN: clang -fsyntax-only -verify %s

@interface A
 -(int) x;
@property (readonly) int x;
@property int ok;
@end

void f0(A *a) {
  a.x = 10;  // expected-error {{assigning to property with 'readonly' attribute not allowed}}
  a.ok = 20;
}

