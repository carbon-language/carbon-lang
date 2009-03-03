// RUN: clang -fsyntax-only -arch x86_64 -verify %s 

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
  // No error is produced with compiling for -arch x86_64 (or "non-fragile" ABI)
  @synthesize sdkPath;
@end

