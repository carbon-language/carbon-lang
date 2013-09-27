; RUN: opt -tbaa -sink -S < %s | FileCheck %s

; CHECK: a:
; CHECK:   %f = load float* %p, !tbaa [[TAGA:!.*]]
; CHECK:   store float %f, float* %q

define void @foo(float* %p, i1 %c, float* %q, float* %r) {
  %f = load float* %p, !tbaa !0
  store float 0.0, float* %r, !tbaa !1
  br i1 %c, label %a, label %b
a:
  store float %f, float* %q
  br label %b
b:
  ret void
}

; CHECK: [[TAGA]] = metadata !{metadata [[TYPEA:!.*]], metadata [[TYPEA]], i64 0}
; CHECK: [[TYPEA]] = metadata !{metadata !"A", metadata !{{.*}}}
!0 = metadata !{metadata !3, metadata !3, i64 0}
!1 = metadata !{metadata !4, metadata !4, i64 0}
!2 = metadata !{metadata !"test"}
!3 = metadata !{metadata !"A", metadata !2}
!4 = metadata !{metadata !"B", metadata !2}
