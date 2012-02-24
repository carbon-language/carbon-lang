; RUN: llc < %s -march=ppc64 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define void @test1() {
entry:
  %call.i75 = call zeroext i8 undef(i8* undef, i8 zeroext 10)
  unreachable
}

; CHECK: @test1
; CHECK: ld 11, 0(3)
; CHECK: ld 2, 8(3)
; CHECK: bctrl
; CHECK: ld 2, 40(1)

