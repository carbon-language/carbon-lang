; RUN: llc < %s -march=x86 -x86-asm-syntax=intel | FileCheck %s

define i32 @test1(i32 %A, i32 %B) {
  %tmp1 = shl i32 %A, 2
  %tmp3 = add i32 %B, -5
  %tmp4 = add i32 %tmp3, %tmp1
; The above computation of %tmp4 should match a single lea, without using
; actual add instructions.
; CHECK-NOT: add
; CHECK: lea {{[A-Z]+}}, DWORD PTR [{{[A-Z]+}} + 4*{{[A-Z]+}} - 5]

  ret i32 %tmp4
}


