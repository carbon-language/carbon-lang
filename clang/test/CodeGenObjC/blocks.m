// RUN: clang -emit-llvm -fblocks -S -o - %s
// rdar://6676764

struct S {
  void (^F)(struct S*);
} P;


@interface T

  - (int)foo: (T (^)(T*)) x;
@end

void foo(T *P) {
 [P foo: 0];
}

@interface A 
-(void) im0;
@end

@interface B : A @end
@implementation B
-(void) im1 {
  ^(void) { [super im0]; }();
}
@end
