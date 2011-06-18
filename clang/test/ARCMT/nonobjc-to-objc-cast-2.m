// RUN: %clang_cc1 -arcmt-check -verify -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi %s

typedef int BOOL;
typedef const struct __CFString * CFStringRef;

@class NSString;

void f(BOOL b) {
  CFStringRef cfstr;
  NSString *str = (NSString *)cfstr; // expected-error {{cast of C pointer type 'CFStringRef' (aka 'const struct __CFString *') to Objective-C pointer type 'NSString *' requires a bridged cast}} \
    // expected-note{{use __bridge to convert directly (no change in ownership)}} \
    // expected-note{{use __bridge_transfer to transfer ownership of a +1 'CFStringRef' (aka 'const struct __CFString *') into ARC}}
  void *vp = str;  // expected-error {{disallowed}}
}
