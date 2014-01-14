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

; CHECK-LABEL: test5
; CHECK: ret
define float @test5(float %p) #0 {
entry:
  %cmp = fcmp oeq float %p, 0.000000e+00
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %cmp1 = fcmp ogt float %p, 0.000000e+00
  %cond = select i1 %cmp1, float 1.000000e+00, float -1.000000e+00
  br label %return

return:                                           ; preds = %if.end, %entry
  %retval.0 = phi float [ %cond, %if.end ], [ %p, %entry ]
  ret float %retval.0
}

; CHECK-LABEL: test6
; CHECK: cmpl
; CHECK-NOT: kmov
; CHECK: ret
define i32 @test6(i32 %a, i32 %b) {
  %cmp = icmp eq i32 %a, %b
  %res = zext i1 %cmp to i32
  ret i32 %res
}

; CHECK-LABEL: test7
; CHECK: vucomisd
; CHECK-NOT: kmov
; CHECK: ret
define i32 @test7(double %x, double %y) #2 {
entry:
  %0 = fcmp one double %x, %y
  %or = zext i1 %0 to i32
  ret i32 %or
}

