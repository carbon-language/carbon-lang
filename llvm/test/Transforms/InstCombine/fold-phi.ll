; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK: no_crash
define float @no_crash(float %a) nounwind {
entry:
  br label %for.body

for.body:
  %sum.057 = phi float [ 0.000000e+00, %entry ], [ %add5, %bb0 ]
  %add5 = fadd float %sum.057, %a    ; PR14592
  br i1 undef, label %bb0, label %end

bb0:
  br label %for.body

end:
  ret float %add5
}

; CHECK: fold_phi
define float @fold_phi(float %a) nounwind {
entry:
  br label %for.body

for.body:
; CHECK: phi float
; CHECK-NEXT: br i1 undef
  %sum.057 = phi float [ 0.000000e+00, %entry ], [ %add5, %bb0 ]
  %add5 = fadd float %sum.057, 1.0 ;; Should be moved to the latch!
  br i1 undef, label %bb0, label %end

; CHECK: bb0:
bb0:
; CHECK: fadd float
  br label %for.body

end:
  ret float %add5
}
