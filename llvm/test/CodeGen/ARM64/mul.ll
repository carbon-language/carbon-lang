; RUN: llc < %s -march=arm64 | FileCheck %s

; rdar://9296808
; rdar://9349137

define i128 @t1(i64 %a, i64 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t1:
; CHECK: mul {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: umulh {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  %tmp1 = zext i64 %a to i128
  %tmp2 = zext i64 %b to i128
  %tmp3 = mul i128 %tmp1, %tmp2
  ret i128 %tmp3
}

define i128 @t2(i64 %a, i64 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t2:
; CHECK: mul {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: smulh {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  %tmp1 = sext i64 %a to i128
  %tmp2 = sext i64 %b to i128
  %tmp3 = mul i128 %tmp1, %tmp2
  ret i128 %tmp3
}

define i64 @t3(i32 %a, i32 %b) nounwind {
entry:
; CHECK-LABEL: t3:
; CHECK: umull {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  %tmp1 = zext i32 %a to i64
  %tmp2 = zext i32 %b to i64
  %tmp3 = mul i64 %tmp1, %tmp2
  ret i64 %tmp3
}

define i64 @t4(i32 %a, i32 %b) nounwind {
entry:
; CHECK-LABEL: t4:
; CHECK: smull {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  %tmp1 = sext i32 %a to i64
  %tmp2 = sext i32 %b to i64
  %tmp3 = mul i64 %tmp1, %tmp2
  ret i64 %tmp3
}

define i64 @t5(i32 %a, i32 %b, i64 %c) nounwind {
entry:
; CHECK-LABEL: t5:
; CHECK: umaddl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{x[0-9]+}}
  %tmp1 = zext i32 %a to i64
  %tmp2 = zext i32 %b to i64
  %tmp3 = mul i64 %tmp1, %tmp2
  %tmp4 = add i64 %c, %tmp3
  ret i64 %tmp4
}

define i64 @t6(i32 %a, i32 %b, i64 %c) nounwind {
entry:
; CHECK-LABEL: t6:
; CHECK: smsubl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{x[0-9]+}}
  %tmp1 = sext i32 %a to i64
  %tmp2 = sext i32 %b to i64
  %tmp3 = mul i64 %tmp1, %tmp2
  %tmp4 = sub i64 %c, %tmp3
  ret i64 %tmp4
}

define i64 @t7(i32 %a, i32 %b) nounwind {
entry:
; CHECK-LABEL: t7:
; CHECK: umnegl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  %tmp1 = zext i32 %a to i64
  %tmp2 = zext i32 %b to i64
  %tmp3 = mul i64 %tmp1, %tmp2
  %tmp4 = sub i64 0, %tmp3
  ret i64 %tmp4
}

define i64 @t8(i32 %a, i32 %b) nounwind {
entry:
; CHECK-LABEL: t8:
; CHECK: smnegl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  %tmp1 = sext i32 %a to i64
  %tmp2 = sext i32 %b to i64
  %tmp3 = mul i64 %tmp1, %tmp2
  %tmp4 = sub i64 0, %tmp3
  ret i64 %tmp4
}
