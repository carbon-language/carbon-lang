; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {orn\\W*r\[0-9\]*,\\W*r\[0-9\]*,\\W*r\[0-9\]*$} | count 4
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {orn\\W*r\[0-9\],\\W*r\[0-9\],\\W*r\[0-9\],\\W*lsl\\W*#5$} | count 1
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {orn\\W*r\[0-9\],\\W*r\[0-9\],\\W*r\[0-9\],\\W*lsr\\W*#6$} | count 1
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {orn\\W*r\[0-9\],\\W*r\[0-9\],\\W*r\[0-9\],\\W*asr\\W*#7$} | count 1
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {orn\\W*r\[0-9\],\\W*r\[0-9\],\\W*r\[0-9\],\\W*ror\\W*#8$} | count 1

define i32 @f1(i32 %a, i32 %b) {
    %tmp = xor i32 %b, 4294967295
    %tmp1 = or i32 %a, %tmp
    ret i32 %tmp1
}

define i32 @f2(i32 %a, i32 %b) {
    %tmp = xor i32 %b, 4294967295
    %tmp1 = or i32 %tmp, %a
    ret i32 %tmp1
}

define i32 @f3(i32 %a, i32 %b) {
    %tmp = xor i32 4294967295, %b
    %tmp1 = or i32 %a, %tmp
    ret i32 %tmp1
}

define i32 @f4(i32 %a, i32 %b) {
    %tmp = xor i32 4294967295, %b
    %tmp1 = or i32 %tmp, %a
    ret i32 %tmp1
}

define i32 @f5(i32 %a, i32 %b) {
    %tmp = shl i32 %b, 5
    %tmp1 = xor i32 4294967295, %tmp
    %tmp2 = or i32 %a, %tmp1
    ret i32 %tmp2
}

define i32 @f6(i32 %a, i32 %b) {
    %tmp = lshr i32 %b, 6
    %tmp1 = xor i32 4294967295, %tmp
    %tmp2 = or i32 %a, %tmp1
    ret i32 %tmp2
}

define i32 @f7(i32 %a, i32 %b) {
    %tmp = ashr i32 %b, 7
    %tmp1 = xor i32 4294967295, %tmp
    %tmp2 = or i32 %a, %tmp1
    ret i32 %tmp2
}

define i32 @f8(i32 %a, i32 %b) {
    %l8 = shl i32 %a, 24
    %r8 = lshr i32 %a, 8
    %tmp = or i32 %l8, %r8
    %tmp1 = xor i32 4294967295, %tmp
    %tmp2 = or i32 %a, %tmp1
    ret i32 %tmp2
}
