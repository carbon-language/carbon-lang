// RUN: %clang_cc1 -fobjc-arc -fobjc-nonfragile-abi -verify %s

// rdar://problem/9150784
void test(void) {
  __weak id x; // expected-error {{the current deployment target does not support automated __weak references}}
}

@interface A
@property (weak) id testObjectWeakProperty; // expected-note {{declared here}}
@end

@implementation A
// rdar://9605088
@synthesize testObjectWeakProperty; // expected-error {{the current deployment target does not support automated __weak references}}
@end
