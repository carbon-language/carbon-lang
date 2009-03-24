// RUN: clang-cc -fsyntax-only -triple x86_64-apple-darwin9 -verify %s 

typedef signed char BOOL;

@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;

@protocol NSObject
- (BOOL)isEqual:(id)object;
@end

@interface NSObject <NSObject> {}
@end

@interface XCDeviceWillExecuteInfoBaton : NSObject {}
  @property (retain) __attribute__((objc_gc(strong))) NSString *sdkPath;
@end

@implementation XCDeviceWillExecuteInfoBaton
  // Produce an error when compiling for -arch x86_64 (or "non-fragile" ABI)
  @synthesize sdkPath; // expected-error{{instance variable synthesis not yet supported (need to declare 'sdkPath' explicitly)}}
@end

