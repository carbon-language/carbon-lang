// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -working-directory %S with-working-dir.m > %t
// RUN: diff %t %s.result

typedef int BOOL;
id IhaveSideEffect();

@protocol NSObject
- (BOOL)isEqual:(id)object;
- (id)retain;
- (oneway void)release;
- (id)something;
@end

@interface NSObject <NSObject> {}
@end

@interface Foo : NSObject {
  id bar;
}
@property (retain) id bar;
-(id)test:(id)obj;
@end

@implementation Foo

@synthesize bar;

-(id)test:(id)obj {
  id x = self.bar;
  [x retain];
  self.bar = obj;
  if (obj)
    [obj retain];

  [IhaveSideEffect() retain];

  [[self something] retain];

  [[self retain] something];

  // do stuff with x;
  [x release];
  return [self retain];
}
  
@end
