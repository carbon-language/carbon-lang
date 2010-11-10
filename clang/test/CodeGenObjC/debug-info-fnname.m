// RUN: %clang_cc1 -emit-llvm -Os -g %s -o - | FileCheck %s
// Radar 8653152
@interface A {
}
@end


// CHECK: llvm.dbg.lv.-.A.title.
@implementation A
-(int) title {
  int x = 1;
  return x;
}
@end

