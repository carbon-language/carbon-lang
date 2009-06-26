; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {orn\\W*r\[0-9\]*,\\W*r\[0-9\]*,\\W*r\[0-9\]*} | count 4

define i32 @f1(i32 %a, i32 %b) {
    %tmp = xor i32 %b, 4294967295
    %tmp1 = or i32 %a, %tmp
    ret i32 %tmp1
}

define i32 @f2(i32 %a, i32 %b) {
    %tmp = xor i32 %b, 4294967295
    %tmp1 = or i32 %tmp, %a
    ret i32 %tmp1
}

define i32 @f3(i32 %a, i32 %b) {
    %tmp = xor i32 4294967295, %b
    %tmp1 = or i32 %a, %tmp
    ret i32 %tmp1
}

define i32 @f4(i32 %a, i32 %b) {
    %tmp = xor i32 4294967295, %b
    %tmp1 = or i32 %tmp, %a
    ret i32 %tmp1
}
