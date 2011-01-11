; RUN: llc -march=sparc <%s | FileCheck %s


define i32 @test_addx(i64 %a, i64 %b, i64 %c) nounwind readnone noinline {
entry:
; CHECK: addcc
; CHECK-NOT: subcc
; CHECK: addx
  %0 = add i64 %a, %b
  %1 = icmp ugt i64 %0, %c
  %2 = zext i1 %1 to i32
  ret i32 %2
}


define i32 @test_select_int_icc(i32 %a, i32 %b, i32 %c) nounwind readnone noinline {
entry:
; CHECK: test_select_int_icc
; CHECK: subcc
; CHECK: be
  %0 = icmp eq i32 %a, 0
  %1 = select i1 %0, i32 %b, i32 %c
  ret i32 %1
}


define float @test_select_fp_icc(i32 %a, float %f1, float %f2) nounwind readnone noinline {
entry:
; CHECK: test_select_fp_icc
; CHECK: subcc
; CHECK: be
  %0 = icmp eq i32 %a, 0
  %1 = select i1 %0, float %f1, float %f2
  ret float %1
}

define double @test_select_dfp_icc(i32 %a, double %f1, double %f2) nounwind readnone noinline {
entry:
; CHECK: test_select_fp_icc
; CHECK: subcc
; CHECK: be
  %0 = icmp eq i32 %a, 0
  %1 = select i1 %0, double %f1, double %f2
  ret double %1
}

define i32 @test_select_int_fcc(float %f, i32 %a, i32 %b) nounwind readnone noinline {
entry:
;CHECK: test_select_int_fcc
;CHECK: fcmps
;CHECK: fbne
  %0 = fcmp une float %f, 0.000000e+00
  %a.b = select i1 %0, i32 %a, i32 %b
  ret i32 %a.b
}


define float @test_select_fp_fcc(float %f, float %f1, float %f2) nounwind readnone noinline {
entry:
;CHECK: test_select_fp_fcc
;CHECK: fcmps
;CHECK: fbne
  %0 = fcmp une float %f, 0.000000e+00
  %1 = select i1 %0, float %f1, float %f2
  ret float %1
}

define double @test_select_dfp_fcc(double %f, double %f1, double %f2) nounwind readnone noinline {
entry:
;CHECK: test_select_dfp_fcc
;CHECK: fcmpd
;CHECK: fbne
  %0 = fcmp une double %f, 0.000000e+00
  %1 = select i1 %0, double %f1, double %f2
  ret double %1
}
