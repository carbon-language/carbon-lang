; RUN: not llvm-as < %s |& FileCheck %s

define void @foo(i32 %i, float %f, <2 x float> %g) {
  %s = add i32 %i, %i, !fpmath !0
; CHECK: fpmath requires a floating point result!
  %t = fadd float %f, %f, !fpmath !1
; CHECK: fpmath takes one operand!
  %u = fadd float %f, %f, !fpmath !2
; CHECK: fpmath takes one operand!
  %v = fadd float %f, %f, !fpmath !3
; CHECK: fpmath ULPs not a floating point number!
  %w = fadd float %f, %f, !fpmath !0
; Above line is correct.
  %w2 = fadd <2 x float> %g, %g, !fpmath !0
; Above line is correct.
  %x = fadd float %f, %f, !fpmath !4
; CHECK: fpmath ULPs is negative!
  %y = fadd float %f, %f, !fpmath !5
; CHECK: fpmath ULPs is negative!
  %z = fadd float %f, %f, !fpmath !6
; CHECK: fpmath ULPs not a normal number!
  ret void
}

!0 = metadata !{ float 1.0 }
!1 = metadata !{ }
!2 = metadata !{ float 1.0, float 1.0 }
!3 = metadata !{ i32 1 }
!4 = metadata !{ float -1.0 }
!5 = metadata !{ float -0.0 }
!6 = metadata !{ float 0x7FFFFFFF00000000 }
