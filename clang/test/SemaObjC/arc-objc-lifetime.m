// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -verify %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -verify %s
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
      (__bridge CFStringRef) (__strong NSString *)CFBridgingRelease(); // expected-error {{explicit ownership qualifier on cast result would have no effect}}

    myString =
      (__bridge CFStringRef) (__autoreleasing PNSString) CFBridgingRelease(); // expected-error {{explicit ownership qualifier on cast result would have no effect}}
    myString =
      (__bridge CFStringRef) (AUTORELEASEPNSString) CFBridgingRelease(); // OK
    myString =
      (__bridge CFStringRef) (typeof(__strong NSString *)) CFBridgingRelease(); // expected-error {{explicit ownership qualifier on cast result would have no effect}}
    return myString;
}

- (void)decodeValueOfObjCType:(const char *)type at:(void *)addr {
        __autoreleasing id *stuff = (__autoreleasing id *)addr;
}
@end
