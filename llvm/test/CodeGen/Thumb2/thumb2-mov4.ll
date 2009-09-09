; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {movw\\W*r\[0-9\],\\W*#\[0-9\]*} | grep {#65535} | count 1

define i32 @f6(i32 %a) {
    %tmp = add i32 0, 65535
    ret i32 %tmp
}
