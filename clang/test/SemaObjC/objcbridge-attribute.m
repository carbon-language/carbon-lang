// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -verify -Wno-objc-root-class %s
// rdar://15454846

typedef struct __CFErrorRef * __attribute__ ((objc_bridge(NSError))) CFErrorRef; // expected-note 2 {{declared here}}

typedef struct __CFMyColor  * __attribute__((objc_bridge(12))) CFMyColorRef; // expected-error {{parameter of 'objc_bridge' attribute must be a single name of an Objective-C class}}

typedef struct __CFArray *  __attribute__ ((objc_bridge)) CFArrayRef; // expected-error {{parameter of 'objc_bridge' attribute must be a single name of an Objective-C class}}

typedef void *  __attribute__ ((objc_bridge(NSURL))) CFURLRef;  // expected-error {{'objc_bridge' attribute must be applied to a pointer to struct type}}

typedef void * CFStringRef __attribute__ ((objc_bridge(NSString))); // expected-error {{'objc_bridge' attribute must be applied to a pointer to struct type}}

typedef struct __CFLocale * __attribute__((objc_bridge(NSLocale, NSError))) CFLocaleRef;// expected-error {{use of undeclared identifier 'NSError'}}

typedef struct __CFData __attribute__((objc_bridge(NSData))) CFDataRef; // expected-error {{'objc_bridge' attribute must be applied to a pointer to struct type}}

typedef struct __attribute__((objc_bridge(NSDictionary))) __CFDictionary * CFDictionaryRef; // expected-error {{'objc_bridge' attribute must be put on a typedef only}}

typedef struct __CFSetRef * CFSetRef __attribute__((objc_bridge(NSSet)));

typedef union __CFUColor * __attribute__((objc_bridge(NSUColor))) CFUColorRef; // expected-error {{'objc_bridge' attribute must be applied to a pointer to struct type}}

@interface I
{
   __attribute__((objc_bridge(NSError))) void * color; // expected-error {{'objc_bridge' attribute must be put on a typedef only}}

}
@end

@protocol NSTesting @end
@class NSString;

typedef struct __CFError * __attribute__((objc_bridge(NSTesting))) CFTestingRef; // expected-note {{declared here}}

id Test1(CFTestingRef cf) {
  return (NSString *)cf; // expected-error {{CF object of type 'CFTestingRef' (aka 'struct __CFError *') is bridged to 'NSTesting', which is not an Objective-C class}}
}

typedef CFErrorRef CFErrorRef1;

typedef CFErrorRef1 CFErrorRef2;

@interface NSError @end

@interface MyError : NSError
@end

@class NSString;

void Test2(CFErrorRef2 cf, NSError *ns, NSString *str, Class c) {
  (void)(NSString *)cf; // expected-warning {{CFErrorRef bridges to NSError, not NSString}}
  (void)(NSError *)cf; // okay
  (void)(MyError*)cf; // okay,
  (void)(CFErrorRef)ns; // okay
  (void)(CFErrorRef)str;  // expected-warning {{NSString cannot bridge to CFErrorRef}}
  (void)(Class)cf; // expected-warning {{CFErrorRef bridges to NSError, not 'Class'}}
  (void)(CFErrorRef)c; // expected-warning {{'Class' cannot bridge to 'CFErrorRef'}}
}
