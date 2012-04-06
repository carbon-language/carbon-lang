// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -verify -Wno-objc-root-class %s
// rdar://10244607

typedef const struct __CFString * CFStringRef;
@class NSString;

NSString *CFBridgingRelease();

typedef NSString * PNSString;

typedef __autoreleasing NSString * AUTORELEASEPNSString;

@interface I @end

@implementation I
- (CFStringRef)myString
{
    CFStringRef myString =
      (__bridge CFStringRef) (__strong NSString *)CFBridgingRelease(); // expected-error {{explicit ownership qualifier on cast result has no effect}}

    myString =
      (__bridge CFStringRef) (__autoreleasing PNSString) CFBridgingRelease(); // expected-error {{explicit ownership qualifier on cast result has no effect}}
    myString =
      (__bridge CFStringRef) (AUTORELEASEPNSString) CFBridgingRelease(); // OK
    myString =
      (__bridge CFStringRef) (typeof(__strong NSString *)) CFBridgingRelease(); // expected-error {{explicit ownership qualifier on cast result has no effect}}
    return myString;
}

- (void)decodeValueOfObjCType:(const char *)type at:(void *)addr {
        __autoreleasing id *stuff = (__autoreleasing id *)addr;
}
@end

// rdar://problem/10711456
__strong I *__strong test1; // expected-error {{the type 'I *__strong' is already explicitly ownership-qualified}}
__strong I *(__strong test2); // expected-error {{the type 'I *__strong' is already explicitly ownership-qualified}}
__strong I *(__strong (test3)); // expected-error {{the type 'I *__strong' is already explicitly ownership-qualified}}
__unsafe_unretained __typeof__(test3) test4;
typedef __strong I *strong_I;
__unsafe_unretained strong_I test5;
