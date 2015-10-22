; RUN: opt -S -analyze -scalar-evolution < %s | FileCheck %s

!0 = !{i8 0, i8 127}

define void @f0(i8* %len_addr) {
; CHECK-LABEL: Classifying expressions for: @f0
 entry:
  %len = load i8, i8* %len_addr, !range !0
  %len_norange = load i8, i8* %len_addr
; CHECK:  %len = load i8, i8* %len_addr, !range !0
; CHECK-NEXT:  -->  %len U: [0,127) S: [0,127)
; CHECK:  %len_norange = load i8, i8* %len_addr
; CHECK-NEXT:  -->  %len_norange U: full-set S: full-set

  %t0 = add i8 %len, 1
  %t1 = add i8 %len, 2
; CHECK:  %t0 = add i8 %len, 1
; CHECK-NEXT:  -->  (1 + %len)<nuw><nsw> U: [1,-128) S: [1,-128)
; CHECK:  %t1 = add i8 %len, 2
; CHECK-NEXT:  -->  (2 + %len)<nuw> U: [2,-127) S: [2,-127)

  %t2 = sub i8 %len, 1
  %t3 = sub i8 %len, 2
; CHECK:  %t2 = sub i8 %len, 1
; CHECK-NEXT:  -->  (-1 + %len)<nsw> U: [-1,126) S: [-1,126)
; CHECK:  %t3 = sub i8 %len, 2
; CHECK-NEXT:  -->  (-2 + %len)<nsw> U: [-2,125) S: [-2,125)

  %q0 = add i8 %len_norange, 1
  %q1 = add i8 %len_norange, 2
; CHECK:  %q0 = add i8 %len_norange, 1
; CHECK-NEXT:  -->  (1 + %len_norange) U: full-set S: full-set
; CHECK:  %q1 = add i8 %len_norange, 2
; CHECK-NEXT:  -->  (2 + %len_norange) U: full-set S: full-set

  %q2 = sub i8 %len_norange, 1
  %q3 = sub i8 %len_norange, 2
; CHECK:  %q2 = sub i8 %len_norange, 1
; CHECK-NEXT:  -->  (-1 + %len_norange) U: full-set S: full-set
; CHECK:  %q3 = sub i8 %len_norange, 2
; CHECK-NEXT:  -->  (-2 + %len_norange) U: full-set S: full-set

  ret void
}
