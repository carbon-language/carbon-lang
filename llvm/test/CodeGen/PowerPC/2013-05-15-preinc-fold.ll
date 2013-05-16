; RUN: llc < %s | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i8* @test(i8* %base, i8 %val) {
entry:
  %arrayidx = getelementptr inbounds i8* %base, i32 -1
  store i8 %val, i8* %arrayidx, align 1
  %arrayidx2 = getelementptr inbounds i8* %base, i32 1
  store i8 %val, i8* %arrayidx2, align 1
  ret i8* %arrayidx
}
; CHECK: @test
; CHECK: %entry
; CHECK-NEXT: stbu 4, -1(3)
; CHECK-NEXT: stb 4, 2(3)
; CHECK-NEXT: blr

