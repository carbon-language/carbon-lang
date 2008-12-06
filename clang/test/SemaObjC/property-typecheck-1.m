// RUN: clang -fsyntax-only -verify %s

@interface A
-(float) x;	// expected-note {{declared at}}
@property int x; // expected-error {{type of property 'x' does not match type of accessor 'x'}}
@end

@interface A (Cat)
@property int moo;	// expected-note {{previous definition is here}}
@end

@implementation A (Cat)
-(int) moo {
  return 0;
}
-(void) setMoo: (float) x { // expected-warning {{conflicting types for 'setMoo:'}}
}
@end

