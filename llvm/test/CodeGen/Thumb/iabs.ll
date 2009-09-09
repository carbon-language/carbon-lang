; RUN: llc < %s -march=thumb -stats |& \
; RUN:   grep {4 .*Number of machine instrs printed}

;; Integer absolute value, should produce something as good as:
;; Thumb:
;;   asr r2, r0, #31
;;   add r0, r0, r2
;;   eor r0, r2
;;   bx lr

define i32 @test(i32 %a) {
        %tmp1neg = sub i32 0, %a
        %b = icmp sgt i32 %a, -1
        %abs = select i1 %b, i32 %a, i32 %tmp1neg
        ret i32 %abs
}

