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
      (__bridge CFStringRef) (__strong NSString *)CFBridgingRelease(); // expected-error {{casting expression of type 'NSString *' to type 'NSString *__strong' with qualified lifetimewill not change object lifetime}}

    myString =
      (__bridge CFStringRef) (__autoreleasing PNSString) CFBridgingRelease(); // expected-error {{casting expression of type 'NSString *' to type '__autoreleasing PNSString' (aka 'NSString *__autoreleasing') with qualified lifetimewill not change object}}
    myString =
      (__bridge CFStringRef) (AUTORELEASEPNSString) CFBridgingRelease(); // OK
    return myString;
}

- (void)decodeValueOfObjCType:(const char *)type at:(void *)addr {
        __autoreleasing id *stuff = (__autoreleasing id *)addr;
}
@end
