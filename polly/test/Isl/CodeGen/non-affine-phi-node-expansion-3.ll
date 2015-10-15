; RUN: opt %loadPolly -polly-codegen \
; RUN:     -S < %s | FileCheck %s

define void @foo(float* %A, i1 %cond0, i1 %cond1) {
entry:
  br label %loop

loop:
  %indvar = phi i64 [0, %entry], [%indvar.next, %backedge]
  %val0 = fadd float 1.0, 2.0
  %val1 = fadd float 1.0, 2.0
  %val2 = fadd float 1.0, 2.0
  br i1 %cond0, label %branch1, label %backedge

; CHECK-LABEL: polly.stmt.loop:
; CHECK-NEXT: %polly.subregion.iv = phi i32 [ 0, %polly.stmt.loop.entry ]
; CHECK-NEXT: %p_val0 = fadd float 1.000000e+00, 2.000000e+00
; CHECK-NEXT: %p_val1 = fadd float 1.000000e+00, 2.000000e+00
; CHECK-NEXT: %p_val2 = fadd float 1.000000e+00, 2.000000e+00
; CHECK-NEXT: store float %p_val0, float* %merge.phiops
; CHECK-NEXT: store float %p_val1, float* %val1.s2a
; CHECK-NEXT: store float %p_val2, float* %val2.s2a

; FIXME -> The last two writes are not really needed and can be dropped if the
;          incoming block of the PHI and the value that is used share the same
;          non-affine region.

branch1:
  br i1 %cond1, label %branch2, label %backedge

; CHECK-LABEL: polly.stmt.branch1:
; CHECK-NEXT:    store float %p_val1, float* %merge.phiops

branch2:
  br label %backedge

; CHECK-LABEL: polly.stmt.branch2:
; CHECK-NEXT:    store float %p_val2, float* %merge.phiops

backedge:
  %merge = phi float [%val0, %loop], [%val1, %branch1], [%val2, %branch2]
  %indvar.next = add i64 %indvar, 1
  store float %merge, float* %A
  %cmp = icmp sle i64 %indvar.next, 100
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
