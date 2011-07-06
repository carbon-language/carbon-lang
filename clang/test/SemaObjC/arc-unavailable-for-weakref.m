// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-nonfragile-abi -fobjc-runtime-has-weak -fsyntax-only -fobjc-arc -verify %s
// rdar://9693477

__attribute__((objc_arc_weak_reference_unavailable))
@interface NSOptOut1072  // expected-note {{class is declared here}}
@end

@interface sub : NSOptOut1072 @end // expected-note 2 {{class is declared here}}

int main() {
  __weak sub *w2; // expected-error {{class is incompatible with __weak references}}

  __weak NSOptOut1072 *ns1; // expected-error {{class is incompatible with __weak references}}

  id obj;

  ns1 = (__weak sub *)obj; // expected-error {{class is incompatible with __weak references}}
}
