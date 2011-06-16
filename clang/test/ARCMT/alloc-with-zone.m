// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result

#include "Common.h"

@interface Foo : NSObject <NSCopying, NSMutableCopying> {
  id bar;
}
@property (retain) id bar;
-(id)test:(NSZone *)z;
@end

@implementation Foo

@synthesize bar;

+(id)class_test:(NSZone *)z {
  return [self allocWithZone:z];
}

-(id)test:(NSZone *)z {
  NSZone *z2 = [self zone], *z3 = z2;
  NSZone *z4 = z3;

  Foo *foo1 = [[Foo allocWithZone:[self zone]] init];
  Foo *foo2 = [[Foo allocWithZone:[super zone]] init];
  Foo *foo3 = [[Foo allocWithZone:z] init];

  Foo *foo4 = [[Foo allocWithZone:z2] init];
  Foo *foo5 = [[Foo allocWithZone:z3] init];
  Foo *foo6 = [[Foo allocWithZone:z4] init];

  foo1 = [foo1 copyWithZone:[self zone]];
  foo2 = [foo1 copyWithZone:[super zone]];
  foo3 = [foo1 copyWithZone:z];
  foo1 = [foo1 mutableCopyWithZone:[self zone]];

  return foo1;
}

@end
