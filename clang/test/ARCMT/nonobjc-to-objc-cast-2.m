// RUN: %clang_cc1 -arcmt-check -verify -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi %s

#include "Common.h"

@interface NSString : NSObject
-(id)string;
-(id)newString;
@end

typedef const struct __CFString * CFStringRef;

void f(BOOL b) {
  CFStringRef cfstr;
  NSString *str = (NSString *)cfstr; // expected-error {{cast of C pointer type 'CFStringRef' (aka 'const struct __CFString *') to Objective-C pointer type 'NSString *' requires a bridged cast}} \
    // expected-note{{use __bridge to convert directly (no change in ownership)}} \
    // expected-note{{use __bridge_transfer to transfer ownership of a +1 'CFStringRef' (aka 'const struct __CFString *') into ARC}}
  void *vp = str;  // expected-error {{disallowed}}
}

void f2(NSString *s) {
  CFStringRef ref;
  ref = [(CFStringRef)[s string] retain]; // expected-error {{cast of Objective-C pointer type 'id' to C pointer type 'CFStringRef' (aka 'const struct __CFString *') requires a bridged cast}} \
    // expected-error {{ bad receiver type 'CFStringRef' (aka 'const struct __CFString *')}} \
    // expected-note{{use __bridge to convert directly (no change in ownership)}} \
    // expected-note{{use __bridge_retained to make an ARC object available as a +1 'CFStringRef' (aka 'const struct __CFString *')}}
}

CFStringRef f3() {
  return (CFStringRef)[[[NSString alloc] init] autorelease]; // expected-error {{it is not safe to cast to 'CFStringRef' the result of 'autorelease' message; a __bridge cast may result in a pointer to a destroyed object and a __bridge_retained may leak the object}} \
    // expected-note {{remove the cast and change return type of function to 'NSString *' to have the object automatically autoreleased}}
}
