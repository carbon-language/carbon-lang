; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {asr\\W*r\[0-9\],\\W*r\[0-9\],\\W*#\[0-9\]*} | grep {#17} | count 1

define i32 @f1(i32 %a) {
    %tmp = ashr i32 %a, 17
    ret i32 %tmp
}
