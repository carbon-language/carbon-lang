; RUN: not llvm-as < %s |& FileCheck %s

define void @foo(i32 %i, float %f, <2 x float> %g) {
  %s = add i32 %i, %i, !fpaccuracy !0
; CHECK: fpaccuracy requires a floating point result!
  %t = fadd float %f, %f, !fpaccuracy !1
; CHECK: fpaccuracy takes one operand!
  %u = fadd float %f, %f, !fpaccuracy !2
; CHECK: fpaccuracy takes one operand!
  %v = fadd float %f, %f, !fpaccuracy !3
; CHECK: fpaccuracy ULPs not a floating point number!
  %w = fadd float %f, %f, !fpaccuracy !0
; Above line is correct.
  %w2 = fadd <2 x float> %g, %g, !fpaccuracy !0
; Above line is correct.
  %x = fadd float %f, %f, !fpaccuracy !4
; CHECK: fpaccuracy ULPs is negative!
  %y = fadd float %f, %f, !fpaccuracy !5
; CHECK: fpaccuracy ULPs is negative!
  %z = fadd float %f, %f, !fpaccuracy !6
; CHECK: fpaccuracy ULPs not a normal number!
  ret void
}

!0 = metadata !{ float 1.0 }
!1 = metadata !{ }
!2 = metadata !{ float 1.0, float 1.0 }
!3 = metadata !{ i32 1 }
!4 = metadata !{ float -1.0 }
!5 = metadata !{ float -0.0 }
!6 = metadata !{ float 0x7FFFFFFF00000000 }
