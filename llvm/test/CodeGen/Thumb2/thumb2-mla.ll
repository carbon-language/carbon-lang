; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {mla\\W*r\[0-9\],\\W*r\[0-9\],\\W*r\[0-9\],\\W*r\[0-9\]} | count 2

define i32 @f1(i32 %a, i32 %b, i32 %c) {
    %tmp1 = mul i32 %a, %b
    %tmp2 = add i32 %c, %tmp1
    ret i32 %tmp2
}

define i32 @f2(i32 %a, i32 %b, i32 %c) {
    %tmp1 = mul i32 %a, %b
    %tmp2 = add i32 %tmp1, %c
    ret i32 %tmp2
}
