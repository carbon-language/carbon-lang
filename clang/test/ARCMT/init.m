// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result

@interface NSObject
-init;
@end

@interface A : NSObject
-init;
-init2;
-foo;
+alloc;
@end

@implementation A
-(id) init {
  [self init];
  id a;
  [a init];
  a = [[A alloc] init];

  return self;
}

-(id) init2 {
  [super init];
  return self;
}

-(id) foo {
  [self init];
  [super init];

  return self;
}
@end
