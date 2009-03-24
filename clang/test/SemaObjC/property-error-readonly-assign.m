// RUN: clang-cc -fsyntax-only -verify %s

@interface A
 -(int) x;
@property (readonly) int x;
@property int ok;
@end

@interface B
 -(void) setOk:(int)arg;
 -(int) x;
 -(int) ok;
@end

void f0(A *a, B* b) {
  a.x = 10;  // expected-error {{assigning to property with 'readonly' attribute not allowed}}
  a.ok = 20;
  b.x = 10;  // expected-error {{setter method is needed to assign to object using property assignment syntax}}
  b.ok = 20;
}

