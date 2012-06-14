; RUN: llc < %s -march=arm -mattr=+v4t | FileCheck %s

;; Integer absolute value, should produce something as good as: ARM:
;;   movs r0, r0
;;   rsbmi r0, r0, #0
;;   bx lr

define i32 @test(i32 %a) {
        %tmp1neg = sub i32 0, %a
        %b = icmp sgt i32 %a, -1
        %abs = select i1 %b, i32 %a, i32 %tmp1neg
        ret i32 %abs
; CHECK:  movs r0, r0
; CHECK:  rsbmi r0, r0, #0
; CHECK:  bx lr
}
