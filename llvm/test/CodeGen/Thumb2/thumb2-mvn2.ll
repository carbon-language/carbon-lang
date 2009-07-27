; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {mvn\\.w\\W*r\[0-9\]*,\\W*r\[0-9\]*$} | count 2
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {mvn\\.w\\W*r\[0-9\],\\W*r\[0-9\],\\W*lsl\\W*#5$} | count 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {mvn\\.w\\W*r\[0-9\],\\W*r\[0-9\],\\W*lsr\\W*#6$} | count 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {mvn\\.w\\W*r\[0-9\],\\W*r\[0-9\],\\W*asr\\W*#7$} | count 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {mvn\\.w\\W*r\[0-9\],\\W*r\[0-9\],\\W*ror\\W*#8$} | count 1

define i32 @f1(i32 %a) {
    %tmp = xor i32 4294967295, %a
    ret i32 %tmp
}

define i32 @f2(i32 %a) {
    %tmp = xor i32 %a, 4294967295
    ret i32 %tmp
}

define i32 @f5(i32 %a) {
    %tmp = shl i32 %a, 5
    %tmp1 = xor i32 %tmp, 4294967295
    ret i32 %tmp1
}

define i32 @f6(i32 %a) {
    %tmp = lshr i32 %a, 6
    %tmp1 = xor i32 %tmp, 4294967295
    ret i32 %tmp1
}

define i32 @f7(i32 %a) {
    %tmp = ashr i32 %a, 7
    %tmp1 = xor i32 %tmp, 4294967295
    ret i32 %tmp1
}

define i32 @f8(i32 %a) {
    %l8 = shl i32 %a, 24
    %r8 = lshr i32 %a, 8
    %tmp = or i32 %l8, %r8
    %tmp1 = xor i32 %tmp, 4294967295
    ret i32 %tmp1
}
