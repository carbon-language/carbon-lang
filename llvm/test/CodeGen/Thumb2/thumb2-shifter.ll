; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep lsl
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep lsr
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep asr
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep ror
; RUN: llc < %s -march=thumb -mattr=+thumb2 | not grep mov

define i32 @t2ADDrs_lsl(i32 %X, i32 %Y) {
        %A = shl i32 %Y, 16
        %B = add i32 %X, %A
        ret i32 %B
}

define i32 @t2ADDrs_lsr(i32 %X, i32 %Y) {
        %A = lshr i32 %Y, 16
        %B = add i32 %X, %A
        ret i32 %B
}

define i32 @t2ADDrs_asr(i32 %X, i32 %Y) {
        %A = ashr i32 %Y, 16
        %B = add i32 %X, %A
        ret i32 %B
}

; i32 ror(n) = (x >> n) | (x << (32 - n))
define i32 @t2ADDrs_ror(i32 %X, i32 %Y) {
        %A = lshr i32 %Y, 16
        %B = shl  i32 %Y, 16
        %C = or   i32 %B, %A
        %R = add  i32 %X, %C
        ret i32 %R
}

define i32 @t2ADDrs_noRegShift(i32 %X, i32 %Y, i8 %sh) {
        %shift.upgrd.1 = zext i8 %sh to i32
        %A = shl i32 %Y, %shift.upgrd.1
        %B = add i32 %X, %A
        ret i32 %B
}

