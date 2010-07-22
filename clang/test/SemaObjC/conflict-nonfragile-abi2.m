// RUN: %clang_cc1 -fobjc-nonfragile-abi -verify -fsyntax-only %s
// rdar : // 8225011

int glob;

@interface I
@property int glob; // expected-note {{property declared here}}
@end

@implementation I
- (int) Meth { return glob; } // expected-warning {{'glob' lookup will access the property ivar in nonfragile-abi2 mode}}
@synthesize glob;
@end
