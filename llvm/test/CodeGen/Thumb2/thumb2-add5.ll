; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {add\\.w\\W*r\[0-9\],\\W*r\[0-9\],\\W*r\[0-9\]$} | count 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {add\\.w\\W*r\[0-9\],\\W*r\[0-9\],\\W*r\[0-9\],\\W*lsl\\W*#5$} | count 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {add\\.w\\W*r\[0-9\],\\W*r\[0-9\],\\W*r\[0-9\],\\W*lsr\\W*#6$} | count 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {add\\.w\\W*r\[0-9\],\\W*r\[0-9\],\\W*r\[0-9\],\\W*asr\\W*#7$} | count 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {add\\.w\\W*r\[0-9\],\\W*r\[0-9\],\\W*r\[0-9\],\\W*ror\\W*#8$} | count 1

define i32 @f1(i32 %a, i32 %b) {
    %tmp = add i32 %a, %b
    ret i32 %tmp
}

define i32 @f2(i32 %a, i32 %b) {
    %tmp = shl i32 %b, 5
    %tmp1 = add i32 %a, %tmp
    ret i32 %tmp1
}

define i32 @f3(i32 %a, i32 %b) {
    %tmp = lshr i32 %b, 6
    %tmp1 = add i32 %a, %tmp
    ret i32 %tmp1
}

define i32 @f4(i32 %a, i32 %b) {
    %tmp = ashr i32 %b, 7
    %tmp1 = add i32 %a, %tmp
    ret i32 %tmp1
}

define i32 @f5(i32 %a, i32 %b) {
    %l8 = shl i32 %a, 24
    %r8 = lshr i32 %a, 8
    %tmp = or i32 %l8, %r8
    %tmp1 = add i32 %a, %tmp
    ret i32 %tmp1
}
