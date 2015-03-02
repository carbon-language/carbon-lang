// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9.0 -fsyntax-only -fapplication-extension %s -verify
// RUN: %clang_cc1 -triple armv7-apple-ios9.0 -fsyntax-only -fapplication-extension %s -verify

#if __has_feature(attribute_availability_app_extension)
 __attribute__((availability(macosx_app_extension,unavailable)))
 __attribute__((availability(ios_app_extension,unavailable)))
#endif
void f0(int); // expected-note {{'f0' has been explicitly marked unavailable here}}

__attribute__((availability(macosx,unavailable)))
__attribute__((availability(ios,unavailable)))
void f1(int); // expected-note {{'f1' has been explicitly marked unavailable here}}

void test() {
  f0(1); // expected-error {{'f0' is unavailable: not available on}}
  f1(1); // expected-error {{'f1' is unavailable}}
}

