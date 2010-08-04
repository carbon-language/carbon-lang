// RUN: %clang_cc1 -fsyntax-only -verify %s -Wselector -include %S/Inputs/chain-selectors1.h -include %S/Inputs/chain-selectors2.h

// RUN: %clang_cc1 -x objective-c -emit-pch -o %t1 %S/Inputs/chain-selectors1.h
// RUN: %clang_cc1 -x objective-c -emit-pch -o %t2 %S/Inputs/chain-selectors2.h -include-pch %t1 -chained-pch
// RUN: %clang_cc1 -fsyntax-only -verify %s -Wselector -include-pch %t2

@implementation X
-(void)f {}
-(void)f2 {}
-(void)g: (int)p {}
-(void)h: (int)p1 foo: (int)p2 {}
@end

void bar() {
  id a = 0;
  [a nothing]; // expected-warning {{method '-nothing' not found}}
  [a f];
  // FIXME: Can't verify notes in headers
  //[a f2];

  (void)@selector(x); // expected-warning {{unimplemented selector}}
  (void)@selector(y); // expected-warning {{unimplemented selector}}
  (void)@selector(e); // expected-warning {{unimplemented selector}}
}
