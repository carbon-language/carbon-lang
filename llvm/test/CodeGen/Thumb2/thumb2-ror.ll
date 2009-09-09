; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {ror\\.w\\W*r\[0-9\]*,\\W*r\[0-9\]*,\\W*#\[0-9\]*} | grep 22 | count 1

define i32 @f1(i32 %a) {
    %l8 = shl i32 %a, 10
    %r8 = lshr i32 %a, 22
    %tmp = or i32 %l8, %r8
    ret i32 %tmp
}
