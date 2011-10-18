; RUN: llc < %s -mtriple=thumbv6-apple-darwin | FileCheck %s

; Normal load from SP
define void @SP(i32 %i) nounwind uwtable ssp {
entry:
; CHECK: @SP
; CHECK: push	{r7, lr}
; CHECK-NEXT: mov r7, sp
; CHECK-NEXT: sub sp, #4
; CHECK-NEXT: mov r1, sp 
; CHECK-NEXT: str r0, [r1]
; CHECK-NEXT: mov r0, sp
; CHECK-NEXT: blx _SP_
; CHECK-NEXT: add sp, #4
; CHECK-NEXT: pop {r7, pc}
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  call void @SP_(i32* %i.addr)
  ret void
}

declare void @SP_(i32*)

; Dynamic stack realignment
define void @FP(double %a) nounwind uwtable ssp {
entry:
; CHECK: mov r4, sp
; CHECK-NEXT: lsrs r4, r4, #3
; CHECK-NEXT: lsls r4, r4, #3
; CHECK-NEXT: mov sp, r4
; Restore from FP
; CHECK: subs r4, r7, #4
; CHECK: mov sp, r4
  %a.addr = alloca double, align 8
  store double %a, double* %a.addr, align 8
  call void @FP_(double* %a.addr)
  ret void
}

declare void @FP_(double*)
