; RUN: llc -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define signext i32 @foo(i32 signext %a) #0 {
entry:
  %mul = mul nsw i32 %a, %a
  %shr2 = lshr i32 %mul, 5
  ret i32 %shr2

; CHECK-LABEL @foo
; CHECK-NOT: rldicl 3, {{[0-9]+}}, 0, 32
; CHECK: blr
}

define zeroext i32 @test6(i32 zeroext %x) #0 {
entry:
  %and = lshr i32 %x, 16
  %shr = and i32 %and, 255
  %and1 = shl i32 %x, 16
  %shl = and i32 %and1, 16711680
  %or = or i32 %shr, %shl
  ret i32 %or

; CHECK-LABEL @test6
; CHECK-NOT: rldicl 3, {{[0-9]+}}, 0, 32
; CHECK: blr
}

attributes #0 = { nounwind readnone }

