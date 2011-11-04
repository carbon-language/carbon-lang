// RUN: %clang_cc1 -arcmt-check -verify -triple x86_64-apple-darwin10 -fobjc-gc-only %s
// RUN: %clang_cc1 -arcmt-check -verify -triple x86_64-apple-darwin10 -fobjc-gc-only -x objective-c++ %s

#define CF_AUTOMATED_REFCOUNT_UNAVAILABLE __attribute__((unavailable("not available in automatic reference counting mode")))
typedef const void * CFTypeRef;
CFTypeRef CFMakeCollectable(CFTypeRef cf) CF_AUTOMATED_REFCOUNT_UNAVAILABLE; // expected-note {{unavailable}}

void test1(CFTypeRef *cft) {
  CFTypeRef c = CFMakeCollectable(cft); // expected-error {{CFMakeCollectable will leak the object that it receives in ARC}} \
                // expected-error {{unavailable}}
}
