// RUN: clang-cc -triple i386-apple-darwin9 -emit-llvm -fblocks -o %t %s
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

// RUN: grep 'define internal i32 @"__-\[A im0\]_block_invoke_"' %t
@implementation A
-(void) im0 {
  (void) ^{ return 1; }();
}
@end

@interface B : A @end
@implementation B
-(void) im1 {
  ^(void) { [super im0]; }();
}
@end

// RUN: true
