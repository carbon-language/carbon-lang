; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {addw\\W*r\[0-9\],\\W*r\[0-9\],\\W*#\[0-9\]*} | grep {#4095} | count 1

define i32 @f1(i32 %a) {
    %tmp = add i32 %a, 4095
    ret i32 %tmp
}
