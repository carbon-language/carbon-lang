; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {eor\\W*r\[0-9\],\\W*r\[0-9\],\\W*r\[0-9\]} | count 1

define i32 @f1(i32 %a, i32 %b) {
    %tmp = xor i32 %a, %b
    ret i32 %tmp
}
