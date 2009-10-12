// RUN: clang-cc -triple i386-apple-darwin9 -g -emit-llvm %s -o - | FileCheck %s
// PR4894
//
// This test is actually just making sure we can generate the debug info for the
// return type from im0 without crashing.
// XFAIL

@interface I0 {
  I0 *_iv0;
}
@end
@protocol P0 @end

@interface I1 @end
@implementation I1
- (I0<P0> *) im0 {
// CHECK: @"\01-[I1 im0]"
// CHECK: llvm.dbg.func.start
  return 0;
}
@end

// FIXME: This was another PR4894 test case, which is crashing somewhere
// else. PR5025.
#if 0
typedef const struct objc_selector {
  void *sel_id;
  const char *sel_types;
} *SEL;

@interface I2
+(id) dictionary;
@end

@implementation I3;
+(void) initialize {
  I2 *a0 = [I2 dictionary];
}
@end
#endif
