// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-gc -fsyntax-only -verify %s
// RUN: %clang_cc1 -x objective-c++ -triple i386-apple-darwin9 -fobjc-gc -fsyntax-only -verify %s

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
@property (assign) __strong id NOT;
@property (assign)  id ID;
@property (assign) INTF* AWEAK;
@property (assign) __weak INTF* WI;
@end	

@implementation INTF
@synthesize pweak=IVAR;  // expected-error {{existing ivar 'IVAR' for __weak property 'pweak' must be __weak}}
@synthesize NOT=II; // expected-error {{property 'NOT' must be declared __weak to match existing ivar 'II' with __weak attribute}}
@synthesize WID;
@synthesize ID;
@synthesize AWEAK; // expected-error {{property 'AWEAK' must be declared __weak to match existing ivar 'AWEAK' with __weak attribute}}
@synthesize WI;
@end
