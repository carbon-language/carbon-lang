; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl --show-mc-encoding | FileCheck %s

; CHECK-LABEL: test1
; CHECK: vucomisd {{.*}}encoding: [0x62
define double @test1(double %a, double %b) nounwind {
  %tobool = fcmp une double %a, %b
  br i1 %tobool, label %l1, label %l2

l1:
  %c = fsub double %a, %b
  ret double %c
l2:
  %c1 = fadd double %a, %b
  ret double %c1
}

; CHECK-LABEL: test2
; CHECK: vucomiss {{.*}}encoding: [0x62
define float @test2(float %a, float %b) nounwind {
  %tobool = fcmp olt float %a, %b
  br i1 %tobool, label %l1, label %l2

l1:
  %c = fsub float %a, %b
  ret float %c
l2:
  %c1 = fadd float %a, %b
  ret float %c1
}

; CHECK-LABEL: test3
; CHECK: vcmpeqss
; CHECK: kmov
; CHECK: ret
define i32 @test3(float %a, float %b) {

  %cmp10.i = fcmp oeq float %a, %b
  %conv11.i = zext i1 %cmp10.i to i32
  ret i32 %conv11.i
}

; CHECK-LABEL: test4
; CHECK: kortestw
; CHECK: jne
; CHECK: ret
declare i32 @llvm.x86.avx512.kortestz.w(i16, i16)

define i16 @test4(i16 %a, i16 %b) {
  %kortz = call i32 @llvm.x86.avx512.kortestz.w(i16 %a, i16 %b)
  %t1 = and i32 %kortz, 1
  %res = icmp eq i32 %t1, 0
  br i1 %res, label %A, label %B

 A: ret i16 %a
 B:
 %b1 = add i16 %a, %b
 ret i16 %b1

}