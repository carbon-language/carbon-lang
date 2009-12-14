// RUN: clang -cc1 -triple i386-apple-darwin9 -fobjc-gc -fsyntax-only -verify %s

@interface INTF
{
  id IVAR;
  __weak id II;
  __weak id WID;
  id ID;
  __weak INTF* AWEAK;
  __weak INTF* WI;
}
@property (assign) __weak id pweak;
@property (assign) __weak id WID;
@property (assign) __strong id not;
@property (assign)  id ID;
@property (assign) INTF* AWEAK;
@property (assign) __weak INTF* WI;
@end	

@implementation INTF
@synthesize pweak=IVAR;  // expected-error {{existing ivar 'IVAR' for __weak property 'pweak' must be __weak}}
@synthesize not=II; // expected-error {{existing ivar 'II' for a __strong property 'not' must be garbage collectable}}
@synthesize WID;
@synthesize ID;
@synthesize AWEAK; // expected-error {{existing ivar 'AWEAK' for a __strong property 'AWEAK' must be garbage collectable}}
@synthesize WI;
@end
