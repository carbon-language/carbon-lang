// RUN: %clang_cc1 %s -verify -fsyntax-only -Wspellcheck

@interface B
@property int x;
@end

@interface S : B
@end

// Spell-checking 'undefined' is ok.
undefined var; // expected-warning {{spell-checking initiated}} \
               // expected-error {{unknown type name}}

typedef int super1;
@implementation S
-(void)foo {
  // Spell-checking 'super' is not ok.
  super.x = 0;
  self.x = 0;
}
@end
