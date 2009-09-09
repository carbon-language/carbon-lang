; RUN: llc < %s -march=arm | grep {bic\\W*r\[0-9\]*,\\W*r\[0-9\]*,\\W*r\[0-9\]*} | count 2

define i32 @f1(i32 %a, i32 %b) {
    %tmp = xor i32 %b, 4294967295
    %tmp1 = and i32 %a, %tmp
    ret i32 %tmp1
}

define i32 @f2(i32 %a, i32 %b) {
    %tmp = xor i32 %b, 4294967295
    %tmp1 = and i32 %tmp, %a
    ret i32 %tmp1
}
