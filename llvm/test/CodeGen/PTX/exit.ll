; RUN: llc < %s -march=ptx | FileCheck %s

define ptx_kernel void @t1() {
; CHECK: exit;
; CHECK-NOT: ret;
  ret void
}

define ptx_kernel void @t2(i32* %p, i32 %x) {
  store i32 %x, i32* %p
; CHECK: exit;
; CHECK-NOT: ret;
  ret void
}
