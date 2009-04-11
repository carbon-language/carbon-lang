// RUN: clang-cc -fsyntax-only -verify %s

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
-(void) setMoo: (float) x { // expected-warning {{conflicting parameter types in implementation of 'setMoo:': 'int' vs 'float'}}
}
@end


typedef int T[2];
typedef void (F)(void);

@interface C
@property(assign) T p2;  // expected-error {{property cannot have array or function type 'T'}}

@property(assign) F f2; // expected-error {{property cannot have array or function type 'F'}}

@end


