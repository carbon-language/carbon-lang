// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-nonfragile-abi -fsyntax-only -fobjc-arc -fblocks -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-nonfragile-abi -fsyntax-only -fblocks %s

typedef const void *CFTypeRef;
typedef const struct __CFString *CFStringRef;

@interface NSString
@end

CFTypeRef CFCreateSomething();
CFStringRef CFCreateString();
CFTypeRef CFGetSomething();
CFStringRef CFGetString();

id CreateSomething();
NSString *CreateNSString();

void from_cf() {
  id obj1 = (__bridge_transfer id)CFCreateSomething();
  id obj2 = (__bridge_transfer NSString*)CFCreateString();
  (__bridge int*)CFCreateSomething(); // expected-error{{incompatible types casting 'CFTypeRef' (aka 'const void *') to 'int *' with a __bridge cast}}
  id obj3 = (__bridge id)CFGetSomething();
  id obj4 = (__bridge NSString*)CFGetString();
}

void to_cf(id obj) {
  CFTypeRef cf1 = (__bridge_retained CFTypeRef)CreateSomething();
  CFStringRef cf2 = (__bridge_retained CFStringRef)CreateNSString();
  CFTypeRef cf3 = (__bridge CFTypeRef)CreateSomething();
  CFStringRef cf4 = (__bridge CFStringRef)CreateNSString();

  // rdar://problem/9629566 - temporary workaround
  CFTypeRef cf5 = (__bridge_retain CFTypeRef)CreateSomething(); // expected-error {{unknown cast annotation __bridge_retain; did you mean __bridge_retained?}}
}

void fixits() {
  id obj1 = (id)CFCreateSomething(); // expected-error{{cast of C pointer type 'CFTypeRef' (aka 'const void *') to Objective-C pointer type 'id' requires a bridged cast}} \
  // expected-note{{use __bridge to convert directly (no change in ownership)}} \
  // expected-note{{use __bridge_transfer to transfer ownership of a +1 'CFTypeRef' (aka 'const void *') into ARC}}
  CFTypeRef cf1 = (CFTypeRef)CreateSomething(); // expected-error{{cast of Objective-C pointer type 'id' to C pointer type 'CFTypeRef' (aka 'const void *') requires a bridged cast}} \
  // expected-note{{use __bridge to convert directly (no change in ownership)}} \
  // expected-note{{use __bridge_retained to make an ARC object available as a +1 'CFTypeRef' (aka 'const void *')}}
}
