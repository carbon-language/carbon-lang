// RUN: %clang_cc1 -fobjc-nonfragile-abi -verify -fsyntax-only %s
// rdar : // 8225011

int glob; // expected-note {{global variable declared here}}

@interface I
@property int glob; // expected-note {{property declared here}}
@property int p;
@property int le;
@property int l;
@property int ls;
@property int r;
@end

@implementation I
- (int) Meth { return glob; } // expected-warning {{when default property synthesis is on, 'glob' lookup will access}}
@synthesize glob;
// rdar: // 8248681
- (int) Meth1: (int) p {
  extern int le;
  int l = 1;
  static int ls;
  register int r;
  p = le + ls + r;
  return l;
}
@dynamic p;
@dynamic le;
@dynamic l;
@dynamic ls;
@dynamic r;
@end


