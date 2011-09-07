// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-gc -fsyntax-only -verify %s
// RUN: %clang_cc1 -x objective-c++ -triple i386-apple-darwin9 -fobjc-gc -fsyntax-only -verify %s

@interface INTF
{
  id IVAR; // expected-note {{ivar is declared here}}
  __weak id II;
  __weak id WID;
  id ID;
  __weak INTF* AWEAK;
  __weak INTF* WI;
}
@property (assign) __weak id pweak;
@property (assign) __weak id WID;
@property (assign) __strong id NOT;
@property (assign)  id ID;
@property (assign) INTF* AWEAK;
@property (assign) __weak INTF* WI;
@end	

@implementation INTF
@synthesize pweak=IVAR;  // expected-error {{existing ivar 'IVAR' for __weak property 'pweak' must be __weak}}
@synthesize NOT=II; // expected-error {{existing ivar 'II' for strong property 'NOT' may not be __weak}}
@synthesize WID;
@synthesize ID;
@synthesize AWEAK; // expected-error {{existing ivar 'AWEAK' for strong property 'AWEAK' may not be __weak}}
@synthesize WI;
@end
