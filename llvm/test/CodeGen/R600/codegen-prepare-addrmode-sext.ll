; RUN: opt -codegenprepare -S -o - %s | FileCheck %s

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "r600--"

; XFAIL: *

; CHECK-LABEL: @test
; CHECK: mul
; CHECK-NEXT: sext
define void @test(i8 addrspace(1)* nocapture readonly %in, i32 %a, i8 %b) {
entry:
  %0 = mul nsw i32 %a, 3
  %1 = sext i32 %0 to i64
  %2 = getelementptr i8 addrspace(1)* %in, i64 %1
  store i8 %b, i8 addrspace(1)* %2
  ret void
}
