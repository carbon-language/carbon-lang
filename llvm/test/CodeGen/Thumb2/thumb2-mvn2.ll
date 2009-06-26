; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {mvn\\W*r\[0-9\]*,\\W*r\[0-9\]*} | count 2

define i32 @f1(i32 %a) {
    %tmp = xor i32 4294967295, %a
    ret i32 %tmp
}

define i32 @f2(i32 %a) {
    %tmp = xor i32 %a, 4294967295
    ret i32 %tmp
}
