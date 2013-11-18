; RUN: llc -march=mips -mattr=-fp64 < %s | FileCheck -check-prefix=CHECK-FP32 %s
; RUN: llc -march=mips -mattr=+fp64 < %s | FileCheck -check-prefix=CHECK-FP64 %s

; This test case is a simplified version of an llvm-stress generated test with
; seed=3718491962.
; It originally failed on MIPS32 with FP64 with the following error:
;     LLVM ERROR: ran out of registers during register allocation
; This was caused by impossible register class restrictions caused by the use
; of BuildPairF64 instead of BuildPairF64_64.

define void @autogen_SD3718491962() {
BB:
  ; CHECK-FP32: mtc1 $zero, $f{{[0-3]*[02468]}}
  ; CHECK-FP32: mtc1 $zero, $f{{[0-3]*[13579]}}

  ; CHECK-FP64: mtc1 $zero, $f{{[0-9]+}}
  ; CHECK-FP64-NOT: mtc1 $zero,
  ; FIXME: A redundant mthc1 is currently emitted. Add a -NOT when it is
  ;        eliminated

  %Cmp = fcmp ule double 0.000000e+00, undef
  %Cmp11 = fcmp ueq double 0xFDBD965CF1BB7FDA, undef
  br label %CF88

CF88:                                             ; preds = %CF86
  %Sl18 = select i1 %Cmp, i1 %Cmp11, i1 %Cmp
  br i1 %Sl18, label %CF88, label %CF85

CF85:                                             ; preds = %CF88
  ret void
}
