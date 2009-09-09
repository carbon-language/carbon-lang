; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {teq\\.w\\W*r\[0-9\],\\W*r\[0-9\]$} | count 4
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {teq\\.w\\W*r\[0-9\],\\W*r\[0-9\],\\W*lsl\\W*#5$} | count 1
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {teq\\.w\\W*r\[0-9\],\\W*r\[0-9\],\\W*lsr\\W*#6$} | count 1
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {teq\\.w\\W*r\[0-9\],\\W*r\[0-9\],\\W*asr\\W*#7$} | count 1
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {teq\\.w\\W*r\[0-9\],\\W*r\[0-9\],\\W*ror\\W*#8$} | count 1

define i1 @f1(i32 %a, i32 %b) {
    %tmp = xor i32 %a, %b
    %tmp1 = icmp ne i32 %tmp, 0
    ret i1 %tmp1
}

define i1 @f2(i32 %a, i32 %b) {
    %tmp = xor i32 %a, %b
    %tmp1 = icmp eq i32 %tmp, 0
    ret i1 %tmp1
}

define i1 @f3(i32 %a, i32 %b) {
    %tmp = xor i32 %a, %b
    %tmp1 = icmp ne i32 0, %tmp
    ret i1 %tmp1
}

define i1 @f4(i32 %a, i32 %b) {
    %tmp = xor i32 %a, %b
    %tmp1 = icmp eq i32 0, %tmp
    ret i1 %tmp1
}

define i1 @f6(i32 %a, i32 %b) {
    %tmp = shl i32 %b, 5
    %tmp1 = xor i32 %a, %tmp
    %tmp2 = icmp eq i32 %tmp1, 0
    ret i1 %tmp2
}

define i1 @f7(i32 %a, i32 %b) {
    %tmp = lshr i32 %b, 6
    %tmp1 = xor i32 %a, %tmp
    %tmp2 = icmp eq i32 %tmp1, 0
    ret i1 %tmp2
}

define i1 @f8(i32 %a, i32 %b) {
    %tmp = ashr i32 %b, 7
    %tmp1 = xor i32 %a, %tmp
    %tmp2 = icmp eq i32 %tmp1, 0
    ret i1 %tmp2
}

define i1 @f9(i32 %a, i32 %b) {
    %l8 = shl i32 %a, 24
    %r8 = lshr i32 %a, 8
    %tmp = or i32 %l8, %r8
    %tmp1 = xor i32 %a, %tmp
    %tmp2 = icmp eq i32 %tmp1, 0
    ret i1 %tmp2
}
