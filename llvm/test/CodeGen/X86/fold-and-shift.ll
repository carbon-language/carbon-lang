; RUN: llc < %s -march=x86 | FileCheck %s

define i32 @t1(i8* %X, i32 %i) {
; CHECK: t1:
; CHECK-NOT: and
; CHECK: movzbl
; CHECK: movl (%{{...}},%{{...}},4),
; CHECK: ret

entry:
  %tmp2 = shl i32 %i, 2
  %tmp4 = and i32 %tmp2, 1020
  %tmp7 = getelementptr i8* %X, i32 %tmp4
  %tmp78 = bitcast i8* %tmp7 to i32*
  %tmp9 = load i32* %tmp78
  ret i32 %tmp9
}

define i32 @t2(i16* %X, i32 %i) {
; CHECK: t2:
; CHECK-NOT: and
; CHECK: movzwl
; CHECK: movl (%{{...}},%{{...}},4),
; CHECK: ret

entry:
  %tmp2 = shl i32 %i, 1
  %tmp4 = and i32 %tmp2, 131070
  %tmp7 = getelementptr i16* %X, i32 %tmp4
  %tmp78 = bitcast i16* %tmp7 to i32*
  %tmp9 = load i32* %tmp78
  ret i32 %tmp9
}
