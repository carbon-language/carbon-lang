; RUN: llc < %s -mtriple thumbv7-apple-ios5.0.0 | FileCheck %s
; PR11107

%struct.huffcodetab = type { i32, i32, i32*, i8* }

@ht = external global [34 x %struct.huffcodetab]

define i32 @func(i32 %table_select, i32 %x, i32 %y, i32* %code) nounwind {
entry:
; CHECK:      BB#0:
; CHECK:        movs.w
; CHECK-NEXT:   it    mi
; CHECK-NEXT:   rsbmi
; CHECK-NEXT:   movs.w
; CHECK-NEXT:   it    mi
; CHECK-NEXT:   rsbmi
  %cmp.i = icmp sgt i32 %x, 0
  %mul.i = sub i32 0, %x
  %mul.i6 = select i1 %cmp.i, i32 %x, i32 %mul.i
  %tmp = select i1 %cmp.i, i32 0, i32 1
  %cmp.i1 = icmp sgt i32 %y, 0
  %mul.i3 = sub i32 0, %y
  %mul.i38 = select i1 %cmp.i1, i32 %y, i32 %mul.i3
  br label %if.then3

if.then3:                                         ; preds = %if.end
  %xlen = getelementptr inbounds [34 x %struct.huffcodetab]* @ht, i32 0, i32 %table_select, i32 0
  %tmp2 = load i32* %xlen, align 4, !tbaa !0
  %sub = add nsw i32 %mul.i6, -15
  %cmp4 = icmp sgt i32 %mul.i6, 14
  %mul.i7 = select i1 %cmp4, i32 15, i32 %mul.i6
  %sub9 = add nsw i32 %mul.i38, -15
  %cmp7 = icmp sgt i32 %mul.i38, 14
  %mul.i39 = select i1 %cmp7, i32 15, i32 %mul.i38
  %mul = shl nsw i32 %mul.i7, 4
  %add = add nsw i32 %mul, %mul.i39
  %table = getelementptr inbounds [34 x %struct.huffcodetab]* @ht, i32 0, i32 %table_select, i32 2
  %tmp3 = load i32** %table, align 4, !tbaa !3
  %arrayidx11 = getelementptr inbounds i32* %tmp3, i32 %add
  %tmp4 = load i32* %arrayidx11, align 4, !tbaa !4
  store i32 %tmp4, i32* %code, align 4, !tbaa !0
  ret i32 42
}

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
!3 = metadata !{metadata !"any pointer", metadata !1}
!4 = metadata !{metadata !"long", metadata !1}
