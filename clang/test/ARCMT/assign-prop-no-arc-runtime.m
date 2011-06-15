// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fsyntax-only -fobjc-arc -fobjc-no-arc-runtime -x objective-c %s.result
// RUN: arcmt-test --args -arch x86_64 %s -D__IPHONE_OS_VERSION_MIN_REQUIRED=40300 > %t
// RUN: diff %t %s.result
// RUN: arcmt-test --args -arch x86_64 %s -miphoneos-version-min=4.3 > %t
// RUN: diff %t %s.result
// RUN: arcmt-test --args -arch x86_64 %s -mmacosx-version-min=10.6 > %t
// RUN: diff %t %s.result

#include "Common.h"

@interface Foo : NSObject {
  NSObject *x;
}
@property (readonly,assign) id x;
@end

@implementation Foo
@synthesize x;
@end
