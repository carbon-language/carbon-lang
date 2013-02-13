;PR14492 - Tablegen incorrectly converts ARM tLDMIA_UPD pseudo to tLDMIA
;RUN: llc -mtriple=thumbv7 < %s  | FileCheck -check-prefix=EXPECTED %s
;RUN: llc -mtriple=thumbv7 < %s  | FileCheck %s

;EXPECTED: foo:
;CHECK: foo:
define i32 @foo(i32* %a) nounwind optsize {
entry:
  %0 = load i32* %a, align 4, !tbaa !0
  %arrayidx1 = getelementptr inbounds i32* %a, i32 1
  %1 = load i32* %arrayidx1, align 4, !tbaa !0
  %arrayidx2 = getelementptr inbounds i32* %a, i32 2
  %2 = load i32* %arrayidx2, align 4, !tbaa !0
  %add.ptr = getelementptr inbounds i32* %a, i32 3
;Make sure we do not have a duplicated register in the front of the reg list
;EXPECTED:  ldm [[BASE:r[0-9]+]]!, {[[REG:r[0-9]+]], {{r[0-9]+}},
;CHECK-NOT: ldm [[BASE:r[0-9]+]]!, {[[REG:r[0-9]+]], [[REG]],
  tail call void @bar(i32* %add.ptr) nounwind optsize
  %add = add nsw i32 %1, %0
  %add3 = add nsw i32 %add, %2
  ret i32 %add3
}

declare void @bar(i32*) optsize

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
