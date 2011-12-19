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
  id obj1 = (__bridge_transfer id)CFCreateSomething(); // expected-error {{bridge casts will have no effect in non-arc mode}}
  id obj2 = (__bridge_transfer NSString*)CFCreateString(); // expected-error {{bridge casts will have no effect in non-arc mode}}
  (__bridge int*)CFCreateSomething();  // expected-error {{bridge casts will have no effect in non-arc mode}}  \
                                       // expected-warning {{expression result unused}}
  id obj3 = (__bridge id)CFGetSomething(); // expected-error {{bridge casts will have no effect in non-arc mode}}
  id obj4 = (__bridge NSString*)CFGetString(); // expected-error {{bridge casts will have no effect in non-arc mode}}
}

void to_cf(id obj) {
  CFTypeRef cf1 = (__bridge_retained CFTypeRef)CreateSomething(); // expected-error {{bridge casts will have no effect in non-arc mode}}
  CFStringRef cf2 = (__bridge_retained CFStringRef)CreateNSString(); // expected-error {{bridge casts will have no effect in non-arc mode}}
  CFTypeRef cf3 = (__bridge CFTypeRef)CreateSomething(); // expected-error {{bridge casts will have no effect in non-arc mode}}
  CFStringRef cf4 = (__bridge CFStringRef)CreateNSString(); // expected-error {{bridge casts will have no effect in non-arc mode}} 
}

void fixits() {
  id obj1 = (id)CFCreateSomething();
  CFTypeRef cf1 = (CFTypeRef)CreateSomething();
}
