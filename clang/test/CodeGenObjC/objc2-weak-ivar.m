// RUN: clang -triple x86_64-unknown-unknown -fobjc-gc -emit-llvm -o %t %s
@class NSObject;

@interface Foo  {
@public
  __weak NSObject *nsobject;
}
@end

@implementation Foo @end
