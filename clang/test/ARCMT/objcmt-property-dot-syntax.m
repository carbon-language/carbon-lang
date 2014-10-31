// RUN: rm -rf %t
// RUN: %clang_cc1 -objcmt-migrate-property-dot-syntax -mt-migrate-directory %t %s -x objective-c -fobjc-runtime-has-weak -fobjc-arc -triple x86_64-apple-darwin11
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result
// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c -fobjc-runtime-has-weak -fobjc-arc %s.result

// rdar://18498572
@interface NSObject @end

@interface P : NSObject
{
  P* obj;
  int i1, i2, i3;
}
@property int count;
@property (copy) P* PropertyReturnsPObj;
- (P*) MethodReturnsPObj;
@end

P* fun();

@implementation P
- (int) Meth : (P*)array {
  [obj setCount : 100];

  [(P*)0 setCount : [array count]];

  [[obj PropertyReturnsPObj] setCount : [array count]];

  [obj setCount : (i1+i2*i3 - 100)];

  return [obj count] -
         [(P*)0 count] + [array count] +
         [fun() count] - 
         [[obj PropertyReturnsPObj] count] +
         [self->obj count];
}

- (P*) MethodReturnsPObj { return 0; }
@end
