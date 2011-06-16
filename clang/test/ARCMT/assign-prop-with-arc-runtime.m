// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-macosx10.7 -fobjc-nonfragile-abi -fsyntax-only %s > %t
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
