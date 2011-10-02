// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -verify %s

typedef const void *CFTypeRef;
typedef const struct __CFString *CFStringRef;

extern CFStringRef CFMakeString0(void);
extern CFStringRef CFCreateString0(void);
void test0() {
  id x;
  x = (id) CFMakeString0(); // expected-error {{requires a bridged cast}} expected-note {{__bridge to convert directly}} expected-note {{__bridge_transfer to transfer}}
  x = (id) CFCreateString0(); // expected-error {{requires a bridged cast}} expected-note {{__bridge to convert directly}} expected-note {{__bridge_transfer to transfer}}
}

extern CFStringRef CFMakeString1(void) __attribute__((cf_returns_not_retained));
extern CFStringRef CFCreateString1(void) __attribute__((cf_returns_retained));
void test1() {
  id x;
  x = (id) CFMakeString1();
  x = (id) CFCreateString1(); // expected-error {{requires a bridged cast}} expected-note {{__bridge to convert directly}} expected-note {{__bridge_transfer to transfer}}
}
