; RUN: llc < %s -march=sparcv9 | FileCheck %s

; CHECK-LABEL: test1:
; CHECK:        sdivx %o0, %o1, %o2
; CHECK-NEXT:   mulx %o2, %o1, %o1
; CHECK-NEXT:   jmp %o7+8
; CHECK-NEXT:   sub %o0, %o1, %o0

define i64 @test1(i64 %X, i64 %Y) {
        %tmp1 = srem i64 %X, %Y
        ret i64 %tmp1
}

; CHECK-LABEL: test2:
; CHECK:        udivx %o0, %o1, %o2
; CHECK-NEXT:   mulx %o2, %o1, %o1
; CHECK-NEXT:   jmp %o7+8
; CHECK-NEXT:   sub %o0, %o1, %o0

define i64 @test2(i64 %X, i64 %Y) {
        %tmp1 = urem i64 %X, %Y
        ret i64 %tmp1
}
