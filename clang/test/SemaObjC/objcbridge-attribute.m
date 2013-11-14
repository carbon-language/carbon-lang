// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://15454846

typedef struct __CFColor * __attribute__ ((objc_bridge(NSError))) CFColorRef;

typedef struct __CFMyColor  * __attribute__((objc_bridge(12))) CFMyColorRef; // expected-error {{parameter of 'objc_bridge' attribute must be a single name of an Objective-C class}}

typedef struct __CFArray *  __attribute__ ((objc_bridge)) CFArrayRef; // expected-error {{parameter of 'objc_bridge' attribute must be a single name of an Objective-C class}}

typedef void *  __attribute__ ((objc_bridge(NSString))) CFRef; 

typedef void * CFTypeRef __attribute__ ((objc_bridge(NSError)));

typedef struct __CFLocale * __attribute__((objc_bridge(NSString, NSError))) CFLocaleRef;// expected-error {{use of undeclared identifier 'NSError'}}

typedef struct __CFData __attribute__((objc_bridge(NSError))) CFDataRef; // expected-error {{'objc_bridge' attribute must be applied to a pointer type}}

typedef struct __attribute__((objc_bridge(NSError))) __CFDictionary * CFDictionaryRef; // expected-error {{'objc_bridge' attribute must be put on a typedef only}}

typedef struct __CFObject * CFObjectRef __attribute__((objc_bridge(NSError)));

typedef union __CFUColor * __attribute__((objc_bridge(NSError))) CFUColorRef; // expected-error {{'objc_bridge' attribute only applies to structs}}

@interface I
{
   __attribute__((objc_bridge(NSError))) void * color; // expected-error {{'objc_bridge' attribute must be put on a typedef only}}

}
@end
