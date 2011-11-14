; RUN: llc < %s -widen-vmovs -mcpu=cortex-a8 -verify-machineinstrs | FileCheck %s
target triple = "thumbv7-apple-ios"

; The 0.0 constant is loaded from the constant pool and kept in a register.
; CHECK: %entry
; CHECK: vldr s
; The float loop variable is initialized with a vmovs from the constant register.
; The vmovs is first widened to a vmovd, and then converted to a vorr because of the v2f32 vadd.f32.
; CHECK: vorr [[DL:d[0-9]+]], [[DN:d[0-9]+]]
; CHECK: , [[DN]]
; CHECK: %for.body.i
; CHECK: vadd.f32 [[DL]], [[DL]], [[DN]]
;
; This test is verifying:
; - The VMOVS widening is happening.
; - Register liveness is verified.
; - The execution domain switch to vorr works across basic blocks.

define void @Mm() nounwind {
entry:
  br label %for.body4

for.body4:
  br label %for.body.i

for.body.i:
  %tmp3.i = phi float [ 0.000000e+00, %for.body4 ], [ %add.i, %for.body.i ]
  %add.i = fadd float %tmp3.i, 0.000000e+00
  %exitcond.i = icmp eq i32 undef, 41
  br i1 %exitcond.i, label %rInnerproduct.exit, label %for.body.i

rInnerproduct.exit:
  store float %add.i, float* undef, align 4
  br label %for.body4
}
