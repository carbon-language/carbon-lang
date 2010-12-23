// RUN: %clang_cc1 -Wnonfragile-abi2 -fsyntax-only -fobjc-nonfragile-abi2 -fobjc-default-synthesize-properties -verify %s
// rdar://8673791

@interface I {
}

@property int IVAR; // expected-note {{property declared here}}
- (int) OK;
@end

@implementation I
- (int) Meth { return IVAR; } // expected-warning {{direct access of synthesized ivar by using property access 'IVAR'}}
- (int) OK { return self.IVAR; }
@end
