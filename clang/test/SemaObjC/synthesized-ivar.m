// RUN: %clang_cc1 -fsyntax-only -fobjc-nonfragile-abi -verify %s
@interface I
{
}
@property int IP;
@end

@implementation I
@synthesize IP;
- (int) Meth {
   return IP;
}
@end

// rdar://7823675
int f0(I *a) { return a->IP; } // expected-error {{instance variable 'IP' is protected}}
