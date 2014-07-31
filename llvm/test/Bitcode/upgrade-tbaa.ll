; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder < %s -preserve-bc-use-list-order

; Function Attrs: nounwind
define void @_Z4testPiPf(i32* nocapture %pI, float* nocapture %pF) #0 {
entry:
  store i32 0, i32* %pI, align 4, !tbaa !{metadata !"int", metadata !0}
  ; CHECK: store i32 0, i32* %pI, align 4, !tbaa [[TAG_INT:!.*]]
  store float 1.000000e+00, float* %pF, align 4, !tbaa !2
  ; CHECK: store float 1.000000e+00, float* %pF, align 4, !tbaa [[TAG_FLOAT:!.*]]
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = metadata !{metadata !"omnipotent char", metadata !1}
!1 = metadata !{metadata !"Simple C/C++ TBAA"}
!2 = metadata !{metadata !"float", metadata !0}

; CHECK: [[TAG_INT]] = metadata !{metadata [[TYPE_INT:!.*]], metadata [[TYPE_INT]], i64 0}
; CHECK: [[TYPE_INT]] = metadata !{metadata !"int", metadata [[TYPE_CHAR:!.*]]}
; CHECK: [[TYPE_CHAR]] = metadata !{metadata !"omnipotent char", metadata !{{.*}}
; CHECK: [[TAG_FLOAT]] = metadata !{metadata [[TYPE_FLOAT:!.*]], metadata [[TYPE_FLOAT]], i64 0}
; CHECK: [[TYPE_FLOAT]] = metadata !{metadata !"float", metadata [[TYPE_CHAR]]}
