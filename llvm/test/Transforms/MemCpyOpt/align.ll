; RUN: opt < %s -S -memcpyopt | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

; The resulting memset is only 4-byte aligned, despite containing
; a 16-byte alignmed store in the middle.

; CHECK: call void @llvm.memset.p0i8.i64(i8* {{.*}}, i8 0, i64 16, i32 4, i1 false)

define void @foo(i32* %p) {
  %a0 = getelementptr i32* %p, i64 0
  store i32 0, i32* %a0, align 4
  %a1 = getelementptr i32* %p, i64 1
  store i32 0, i32* %a1, align 16
  %a2 = getelementptr i32* %p, i64 2
  store i32 0, i32* %a2, align 4
  %a3 = getelementptr i32* %p, i64 3
  store i32 0, i32* %a3, align 4
  ret void
}
