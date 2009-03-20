// RUN: clang -emit-llvm -o %t %s

@interface B
  +(int) classGetter;
  +(void) setClassGetter:(int) arg;

  -(int) getter;
  -(void) setGetter:(int)arg;
@end

@interface A : B
@end

@implementation A
+(int) classGetter {
  return 0;
}

+(int) classGetter2 {
  super.classGetter = 100;
  return super.classGetter;
}

-(void) method {
  super.getter = 200;
  int x = super.getter;
}
@end

void f0() {
  int l1 = A.classGetter;
  int l2 = [A classGetter2];
}
