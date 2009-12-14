// RUN: clang -cc1 -triple x86_64-apple-darwin9 -fobjc-gc -emit-llvm -o %t %s
@class NSObject;

@interface Foo  {
@public
  __weak NSObject *nsobject;
}
@end

@implementation Foo @end
