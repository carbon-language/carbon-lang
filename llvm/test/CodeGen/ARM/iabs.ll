; RUN: llc < %s -march=arm -stats |& \
; RUN:   grep {3 .*Number of machine instrs printed}

;; Integer absolute value, should produce something as good as: ARM:
;;   add r3, r0, r0, asr #31
;;   eor r0, r3, r0, asr #31
;;   bx lr

define i32 @test(i32 %a) {
        %tmp1neg = sub i32 0, %a
        %b = icmp sgt i32 %a, -1
        %abs = select i1 %b, i32 %a, i32 %tmp1neg
        ret i32 %abs
}

