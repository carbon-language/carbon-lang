; RUN: llc  < %s -march=mipsel -mcpu=4ke | FileCheck %s -check-prefix=CHECK-MIPS32R2
; RUN: llc  < %s -march=mipsel | FileCheck %s -check-prefix=CHECK-MIPS1

@d2 = external global double
@d3 = external global double

define i32 @sel1(i32 %s, i32 %f0, i32 %f1) nounwind readnone {
entry:
; CHECK-MIPS32R2: movn
; CHECK-MIPS1: beq
  %tobool = icmp ne i32 %s, 0
  %cond = select i1 %tobool, i32 %f1, i32 %f0
  ret i32 %cond
}

define float @sel2(i32 %s, float %f0, float %f1) nounwind readnone {
entry:
; CHECK-MIPS32R2: movn.s
; CHECK-MIPS1: beq
  %tobool = icmp ne i32 %s, 0
  %cond = select i1 %tobool, float %f0, float %f1
  ret float %cond
}

define double @sel2_1(i32 %s, double %f0, double %f1) nounwind readnone {
entry:
; CHECK-MIPS32R2: movn.d
; CHECK-MIPS1: beq
  %tobool = icmp ne i32 %s, 0
  %cond = select i1 %tobool, double %f0, double %f1
  ret double %cond
}

define float @sel3(float %f0, float %f1, float %f2, float %f3) nounwind readnone {
entry:
; CHECK-MIPS32R2: c.eq.s
; CHECK-MIPS32R2: movt.s
; CHECK-MIPS1: c.eq.s
; CHECK-MIPS1: bc1f
  %cmp = fcmp oeq float %f2, %f3
  %cond = select i1 %cmp, float %f0, float %f1
  ret float %cond
}

define float @sel4(float %f0, float %f1, float %f2, float %f3) nounwind readnone {
entry:
; CHECK-MIPS32R2: c.olt.s
; CHECK-MIPS32R2: movt.s
; CHECK-MIPS1: c.olt.s
; CHECK-MIPS1: bc1f
  %cmp = fcmp olt float %f2, %f3
  %cond = select i1 %cmp, float %f0, float %f1
  ret float %cond
}

define float @sel5(float %f0, float %f1, float %f2, float %f3) nounwind readnone {
entry:
; CHECK-MIPS32R2: c.ule.s
; CHECK-MIPS32R2: movf.s
; CHECK-MIPS1: c.ule.s
; CHECK-MIPS1: bc1t
  %cmp = fcmp ogt float %f2, %f3
  %cond = select i1 %cmp, float %f0, float %f1
  ret float %cond
}

define double @sel5_1(double %f0, double %f1, float %f2, float %f3) nounwind readnone {
entry:
; CHECK-MIPS32R2: c.ule.s
; CHECK-MIPS32R2: movf.d
; CHECK-MIPS1: c.ule.s
; CHECK-MIPS1: bc1t
  %cmp = fcmp ogt float %f2, %f3
  %cond = select i1 %cmp, double %f0, double %f1
  ret double %cond
}

define double @sel6(double %f0, double %f1, double %f2, double %f3) nounwind readnone {
entry:
; CHECK-MIPS32R2: c.eq.d
; CHECK-MIPS32R2: movt.d
; CHECK-MIPS1: c.eq.d
; CHECK-MIPS1: bc1f
  %cmp = fcmp oeq double %f2, %f3
  %cond = select i1 %cmp, double %f0, double %f1
  ret double %cond
}

define double @sel7(double %f0, double %f1, double %f2, double %f3) nounwind readnone {
entry:
; CHECK-MIPS32R2: c.olt.d
; CHECK-MIPS32R2: movt.d
; CHECK-MIPS1: c.olt.d
; CHECK-MIPS1: bc1f
  %cmp = fcmp olt double %f2, %f3
  %cond = select i1 %cmp, double %f0, double %f1
  ret double %cond
}

define double @sel8(double %f0, double %f1, double %f2, double %f3) nounwind readnone {
entry:
; CHECK-MIPS32R2: c.ule.d
; CHECK-MIPS32R2: movf.d
; CHECK-MIPS1: c.ule.d
; CHECK-MIPS1: bc1t
  %cmp = fcmp ogt double %f2, %f3
  %cond = select i1 %cmp, double %f0, double %f1
  ret double %cond
}

define float @sel8_1(float %f0, float %f1, double %f2, double %f3) nounwind readnone {
entry:
; CHECK-MIPS32R2: c.ule.d
; CHECK-MIPS32R2: movf.s
; CHECK-MIPS1: c.ule.d
; CHECK-MIPS1: bc1t
  %cmp = fcmp ogt double %f2, %f3
  %cond = select i1 %cmp, float %f0, float %f1
  ret float %cond
}

define i32 @sel9(i32 %f0, i32 %f1, float %f2, float %f3) nounwind readnone {
entry:
; CHECK-MIPS32R2: c.eq.s
; CHECK-MIPS32R2: movt
; CHECK-MIPS1: c.eq.s
; CHECK-MIPS1: bc1f
  %cmp = fcmp oeq float %f2, %f3
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

define i32 @sel10(i32 %f0, i32 %f1, float %f2, float %f3) nounwind readnone {
entry:
; CHECK-MIPS32R2: c.olt.s
; CHECK-MIPS32R2: movt
; CHECK-MIPS1: c.olt.s
; CHECK-MIPS1: bc1f
  %cmp = fcmp olt float %f2, %f3
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

define i32 @sel11(i32 %f0, i32 %f1, float %f2, float %f3) nounwind readnone {
entry:
; CHECK-MIPS32R2: c.ule.s
; CHECK-MIPS32R2: movf
; CHECK-MIPS1: c.ule.s
; CHECK-MIPS1: bc1t
  %cmp = fcmp ogt float %f2, %f3
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

define i32 @sel12(i32 %f0, i32 %f1) nounwind readonly {
entry:
; CHECK-MIPS32R2: c.eq.d
; CHECK-MIPS32R2: movt
; CHECK-MIPS1: c.eq.d
; CHECK-MIPS1: bc1f
  %tmp = load double* @d2, align 8, !tbaa !0
  %tmp1 = load double* @d3, align 8, !tbaa !0
  %cmp = fcmp oeq double %tmp, %tmp1
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

define i32 @sel13(i32 %f0, i32 %f1) nounwind readonly {
entry:
; CHECK-MIPS32R2: c.olt.d
; CHECK-MIPS32R2: movt
; CHECK-MIPS1: c.olt.d
; CHECK-MIPS1: bc1f
  %tmp = load double* @d2, align 8, !tbaa !0
  %tmp1 = load double* @d3, align 8, !tbaa !0
  %cmp = fcmp olt double %tmp, %tmp1
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

define i32 @sel14(i32 %f0, i32 %f1) nounwind readonly {
entry:
; CHECK-MIPS32R2: c.ule.d
; CHECK-MIPS32R2: movf
; CHECK-MIPS1: c.ule.d
; CHECK-MIPS1: bc1t
  %tmp = load double* @d2, align 8, !tbaa !0
  %tmp1 = load double* @d3, align 8, !tbaa !0
  %cmp = fcmp ogt double %tmp, %tmp1
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

!0 = metadata !{metadata !"double", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
