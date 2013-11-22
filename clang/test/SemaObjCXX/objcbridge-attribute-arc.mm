// RUN: %clang_cc1 -fsyntax-only -x objective-c++ -fobjc-arc -verify -Wno-objc-root-class %s
// rdar://15454846

typedef struct __attribute__ ((objc_bridge(NSError))) __CFErrorRef * CFErrorRef; // expected-note 7 {{declared here}}

typedef struct __attribute__ ((objc_bridge(MyError))) __CFMyErrorRef * CFMyErrorRef; // expected-note 3 {{declared here}}

typedef struct __attribute__((objc_bridge(12))) __CFMyColor  *CFMyColorRef; // expected-error {{parameter of 'objc_bridge' attribute must be a single name of an Objective-C class}}

typedef struct __attribute__ ((objc_bridge)) __CFArray *CFArrayRef; // expected-error {{'objc_bridge' attribute takes one argument}}

typedef void *  __attribute__ ((objc_bridge(NSURL))) CFURLRef;  // expected-error {{'objc_bridge' attribute only applies to struct, union or class}}

typedef void * CFStringRef __attribute__ ((objc_bridge(NSString))); // expected-error {{'objc_bridge' attribute only applies to struct, union or class}}

typedef struct __attribute__((objc_bridge(NSLocale, NSError))) __CFLocale *CFLocaleRef;// expected-error {{use of undeclared identifier 'NSError'}}

typedef struct __CFData __attribute__((objc_bridge(NSData))) CFDataRef; // expected-error {{'objc_bridge' attribute only applies to struct, union or class}}

typedef struct __attribute__((objc_bridge(NSDictionary))) __CFDictionary * CFDictionaryRef;

typedef struct __CFSetRef * CFSetRef __attribute__((objc_bridge(NSSet))); // expected-error {{'objc_bridge' attribute only applies to struct, union or class}};

typedef union __CFUColor __attribute__((objc_bridge(NSUColor))) * CFUColorRef; // expected-error {{'objc_bridge' attribute only applies to struct, union or class}};

typedef union __CFUColor __attribute__((objc_bridge(NSUColor))) *CFUColor1Ref; // expected-error {{'objc_bridge' attribute only applies to struct, union or class}};

typedef union __attribute__((objc_bridge(NSUColor))) __CFUPrimeColor XXX;
typedef XXX *CFUColor2Ref;

@interface I
{
   __attribute__((objc_bridge(NSError))) void * color; // expected-error {{'objc_bridge' attribute only applies to struct, union or class}};
}
@end

@protocol NSTesting @end
@class NSString;

typedef struct __attribute__((objc_bridge(NSTesting))) __CFError *CFTestingRef; // expected-note {{declared here}}

id Test1(CFTestingRef cf) {
  return (NSString *)cf; // expected-error {{CF object of type 'CFTestingRef' (aka '__CFError *') is bridged to 'NSTesting', which is not an Objective-C class}}
}

typedef CFErrorRef CFErrorRef1;

typedef CFErrorRef1 CFErrorRef2; // expected-note 2 {{declared here}}

@protocol P1 @end
@protocol P2 @end
@protocol P3 @end
@protocol P4 @end
@protocol P5 @end

@interface NSError<P1, P2, P3> @end // expected-note 7 {{declared here}}

@interface MyError : NSError // expected-note 3 {{declared here}}
@end

@interface NSUColor @end

@class NSString;

void Test2(CFErrorRef2 cf, NSError *ns, NSString *str, Class c, CFUColor2Ref cf2) {
  (void)(NSString *)cf; // expected-warning {{'CFErrorRef2' (aka '__CFErrorRef *') bridges to NSError, not 'NSString'}}
  (void)(NSError *)cf; // okay
  (void)(MyError*)cf; // okay,
  (void)(NSUColor *)cf2; // okay
  (void)(CFErrorRef)ns; // okay
  (void)(CFErrorRef)str;  // expected-warning {{'NSString' cannot bridge to 'CFErrorRef' (aka '__CFErrorRef *')}}
  (void)(Class)cf; // expected-warning {{'CFErrorRef2' (aka '__CFErrorRef *') bridges to NSError, not 'Class'}}
  (void)(CFErrorRef)c; // expected-warning {{'Class' cannot bridge to 'CFErrorRef'}}
}


void Test3(CFErrorRef cf, NSError *ns) {
  (void)(id)cf; // okay
 (void)(id<P1, P2>)cf; // okay
 (void)(id<P1, P2, P4>)cf; // expected-warning {{'CFErrorRef' (aka '__CFErrorRef *') bridges to NSError, not 'id<P1,P2,P4>'}}
}

void Test4(CFMyErrorRef cf) {
   (void)(id)cf; // okay
 (void)(id<P1, P2>)cf; // ok
 (void)(id<P1, P2, P3>)cf; // ok
 (void)(id<P2, P3>)cf; // ok
 (void)(id<P1, P2, P4>)cf; // expected-warning {{'CFMyErrorRef' (aka '__CFMyErrorRef *') bridges to MyError, not 'id<P1,P2,P4>'}}
}

void Test5(id<P1, P2, P3> P123, id ID, id<P1, P2, P3, P4> P1234, id<P1, P2> P12, id<P2, P3> P23) {
 (void)(CFErrorRef)ID; // ok
 (void)(CFErrorRef)P123; // ok
 (void)(CFErrorRef)P1234; // ok
 (void)(CFErrorRef)P12; // expected-warning {{'id<P1,P2>' cannot bridge to 'CFErrorRef' (aka '__CFErrorRef *')}}
 (void)(CFErrorRef)P23; // expected-warning {{'id<P2,P3>' cannot bridge to 'CFErrorRef' (aka '__CFErrorRef *')}}
}

void Test6(id<P1, P2, P3> P123, id ID, id<P1, P2, P3, P4> P1234, id<P1, P2> P12, id<P2, P3> P23) {

 (void)(CFMyErrorRef)ID; // ok
 (void)(CFMyErrorRef)P123; // ok
 (void)(CFMyErrorRef)P1234; // ok
 (void)(CFMyErrorRef)P12; // expected-warning {{'id<P1,P2>' cannot bridge to 'CFMyErrorRef' (aka '__CFMyErrorRef *')}}
 (void)(CFMyErrorRef)P23; // expected-warning {{'id<P2,P3>' cannot bridge to 'CFMyErrorRef' (aka '__CFMyErrorRef *')}}
}

typedef struct __attribute__ ((objc_bridge(MyPersonalError))) __CFMyPersonalErrorRef * CFMyPersonalErrorRef;  // expected-note 4 {{declared here}}

@interface MyPersonalError : NSError <P4> // expected-note 4 {{declared here}}
@end

void Test7(id<P1, P2, P3> P123, id ID, id<P1, P2, P3, P4> P1234, id<P1, P2> P12, id<P2, P3> P23) {
 (void)(CFMyPersonalErrorRef)ID; // ok
 (void)(CFMyPersonalErrorRef)P123; // expected-warning {{'id<P1,P2,P3>' cannot bridge to 'CFMyPersonalErrorRef' (aka '__CFMyPersonalErrorRef *')}}
 (void)(CFMyPersonalErrorRef)P1234; // ok
 (void)(CFMyPersonalErrorRef)P12; //  expected-warning {{'id<P1,P2>' cannot bridge to 'CFMyPersonalErrorRef' (aka '__CFMyPersonalErrorRef *')}}
 (void)(CFMyPersonalErrorRef)P23; //  expected-warning {{'id<P2,P3>' cannot bridge to 'CFMyPersonalErrorRef' (aka '__CFMyPersonalErrorRef *')}}
}

void Test8(CFMyPersonalErrorRef cf) {
  (void)(id)cf; // ok
  (void)(id<P1>)cf; // ok
  (void)(id<P1, P2>)cf; // ok
  (void)(id<P1, P2, P3>)cf; // ok
  (void)(id<P1, P2, P3, P4>)cf; // ok
  (void)(id<P1, P2, P3, P4, P5>)cf; // expected-warning {{'CFMyPersonalErrorRef' (aka '__CFMyPersonalErrorRef *') bridges to MyPersonalError, not 'id<P1,P2,P3,P4,P5>'}}
}

void Test9(CFErrorRef2 cf, NSError *ns, NSString *str, Class c, CFUColor2Ref cf2) {
  (void)(__bridge NSString *)cf; // expected-warning {{'CFErrorRef2' (aka '__CFErrorRef *') bridges to NSError, not 'NSString'}}
  (void)(__bridge NSError *)cf; // okay
  (void)(__bridge MyError*)cf; // okay,
  (void)(__bridge NSUColor *)cf2; // okay
  (void)(__bridge CFErrorRef)ns; // okay
  (void)(__bridge CFErrorRef)str;  // expected-warning {{'NSString' cannot bridge to 'CFErrorRef' (aka '__CFErrorRef *')}}
  (void)(__bridge Class)cf; // expected-warning {{'CFErrorRef2' (aka '__CFErrorRef *') bridges to NSError, not 'Class'}}
  (void)(__bridge CFErrorRef)c; // expected-warning {{'__unsafe_unretained Class' cannot bridge to 'CFErrorRef' (aka '__CFErrorRef *')}}
}
