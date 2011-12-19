// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fblocks -verify %s
// rdar://10597832

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
  id obj1 = (__bridge_transfer id)CFCreateSomething(); // expected-warning {{bridge casts will have no effect in non-arc mode and will be ignored}}
  id obj2 = (__bridge_transfer NSString*)CFCreateString(); // expected-warning {{bridge casts will have no effect in non-arc mode and will be ignored}}
  (__bridge int*)CFCreateSomething();  // expected-warning {{bridge casts will have no effect in non-arc mode and will be ignored}}  \
                                       // expected-warning {{expression result unused}}
  id obj3 = (__bridge id)CFGetSomething(); // expected-warning {{bridge casts will have no effect in non-arc mode and will be ignored}}
  id obj4 = (__bridge NSString*)CFGetString(); // expected-warning {{bridge casts will have no effect in non-arc mode and will be ignored}}
}

void to_cf(id obj) {
  CFTypeRef cf1 = (__bridge_retained CFTypeRef)CreateSomething(); // expected-warning {{bridge casts will have no effect in non-arc mode and will be ignored}}
  CFStringRef cf2 = (__bridge_retained CFStringRef)CreateNSString(); // expected-warning {{bridge casts will have no effect in non-arc mode and will be ignored}}
  CFTypeRef cf3 = (__bridge CFTypeRef)CreateSomething(); // expected-warning {{bridge casts will have no effect in non-arc mode and will be ignored}}
  CFStringRef cf4 = (__bridge CFStringRef)CreateNSString(); // expected-warning {{bridge casts will have no effect in non-arc mode and will be ignored}}
}

void fixits() {
  id obj1 = (id)CFCreateSomething();
  CFTypeRef cf1 = (CFTypeRef)CreateSomething();
}
