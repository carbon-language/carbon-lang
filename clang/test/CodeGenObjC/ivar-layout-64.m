// RUNX: llvm-gcc -m64 -fobjc-gc -emit-llvm -S -o %t %s &&
// RUN: clang-cc -triple x86_64-apple-darwin9 -fobjc-gc -emit-llvm -o %t %s &&
// RUN: grep '@"\\01L_OBJC_CLASS_NAME_.*" = internal global .* c"A\\00"' %t &&
// RUN: grep '@"\\01L_OBJC_CLASS_NAME_.*" = internal global .* c"\\11q\\10\\00"' %t &&
// RUN: grep '@"\\01L_OBJC_CLASS_NAME_.*" = internal global .* c"!q\\00"' %t &&
// RUN: true

/*

Here is a handy command for looking at llvm-gcc's output:
llvm-gcc -m64 -fobjc-gc -emit-llvm -S -o - ivar-layout-64.m | \
  grep 'OBJC_CLASS_NAME.* =.*global' | \
  sed -e 's#, section.*# ...#' | \
  sed -e 's#_[0-9]*"#_NNN#' | \
  sort

*/

@interface B @end

@interface A {
  struct s0 {
    int f0;
    int f1;
  } f0;
  id f1;
__weak B *f2;
  int f3 : 5;
  struct s1 {
    int *f0;
    int *f1;
  } f4[2][1];
}
@end

@interface C : A
@property int p3;
@end

@implementation C
@synthesize p3 = _p3;
@end

@interface A()
@property int p0;
@property (assign) __strong id p1;
@property (assign) __weak id p2;
@end

// FIXME: Check layout for this class, once it is clear what the right
// answer is.
@implementation A
@synthesize p0 = _p0;
@synthesize p1 = _p1;
@synthesize p2 = _p2;
@end

@interface D : A
@property int p3;
@end

// FIXME: Check layout for this class, once it is clear what the right
// answer is.
@implementation D
@synthesize p3 = _p3;
@end
