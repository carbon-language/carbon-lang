// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://15454846

typedef struct CGColor * __attribute__ ((objc_bridge(NSError))) CGColorRef;

typedef struct CGColor * __attribute__((objc_bridge(12))) CFMyColorRef; // expected-error {{parameter of 'objc_bridge' attribute must be a single name of an Objective-C class}}

typedef struct S1 *  __attribute__ ((objc_bridge)) CGColorRef1; // expected-error {{parameter of 'objc_bridge' attribute must be a single name of an Objective-C class}}

typedef void *  __attribute__ ((objc_bridge(NSString))) CGColorRef2; 

typedef void * CFTypeRef __attribute__ ((objc_bridge(NSError)));

typedef struct CGColor * __attribute__((objc_bridge(NSString, NSError))) CGColorRefNoNSObject;// expected-error {{use of undeclared identifier 'NSError'}}

typedef struct CGColor __attribute__((objc_bridge(NSError))) CGColorRefNoNSObject2; // expected-error {{'objc_bridge' attribute must be applied to a pointer type}}

typedef struct __attribute__((objc_bridge(NSError))) CFColor * CFColorRefNoNSObject; // expected-error {{'objc_bridge' attribute must be put on a typedef only}}

typedef struct __attribute__((objc_bridge(NSError))) CFColor * CFColorRefNoNSObject1;

typedef union CFUColor * __attribute__((objc_bridge(NSError))) CFColorRefNoNSObject2; // expected-error {{'objc_bridge' attribute only applies to structs}}

@interface I
{
   __attribute__((objc_bridge(NSError))) void * color; // expected-error {{'objc_bridge' attribute must be put on a typedef only}}

}
@end
