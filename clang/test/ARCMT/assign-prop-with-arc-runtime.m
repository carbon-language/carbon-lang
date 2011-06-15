// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -arch x86_64 %s -D__IPHONE_OS_VERSION_MIN_REQUIRED=50000 > %t
// RUN: diff %t %s.result
// RUN: arcmt-test --args -arch x86_64 %s -miphoneos-version-min=5.0 > %t
// RUN: diff %t %s.result
// RUN: arcmt-test --args -arch x86_64 %s -mmacosx-version-min=10.7 > %t
// RUN: diff %t %s.result

#include "Common.h"

@interface Foo : NSObject {
  NSObject *x, *w, *q1, *q2;
  NSObject *z1, *__unsafe_unretained z2;
}
@property (readonly,assign) id x;
@property (assign) id w;
@property (assign) id q1, q2;
@property (assign) id z1, z2;
@end

@implementation Foo
@synthesize x,w,q1,q2,z1,z2;
@end
