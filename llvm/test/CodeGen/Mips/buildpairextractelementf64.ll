; RUN: llc -march=mipsel < %s | FileCheck %s -check-prefix=FP32
; RUN: llc -march=mips  < %s | FileCheck %s -check-prefix=FP32
; RUN: llc -march=mipsel -mattr=+fp64 < %s | FileCheck %s -check-prefix=FP64
; RUN: llc -march=mips -mattr=+fp64 < %s | FileCheck %s -check-prefix=FP64

@a = external global i32

; CHECK-LABEL: f:
; FP32: mtc1
; FP32: mtc1
; FP64-DAG: mtc1
; FP64-DAG: mthc1

define double @f(i32 %a1, double %d) nounwind {
entry:
  store i32 %a1, i32* @a, align 4
  %add = fadd double %d, 2.000000e+00
  ret double %add
}

; CHECK-LABEL: f3:
; FP32: mfc1
; FP32: mfc1
; FP64-DAG: mfc1
; FP64-DAG: mfhc1

define void @f3(double %d, i32 %a1) nounwind {
entry:
  tail call void @f2(i32 %a1, double %d) nounwind
  ret void
}

declare void @f2(i32, double)

