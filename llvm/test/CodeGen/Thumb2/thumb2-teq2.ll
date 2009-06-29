; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {teq\\W*r\[0-9\],\\W*r\[0-9\]} | count 4

define i1 @f1(i32 %a, i32 %b) {
    %tmp = xor i32 %a, %b
    %tmp1 = icmp ne i32 %tmp, 0
    ret i1 %tmp1
}

define i1 @f2(i32 %a, i32 %b) {
    %tmp = xor i32 %a, %b
    %tmp1 = icmp eq i32 %tmp, 0
    ret i1 %tmp1
}

define i1 @f3(i32 %a, i32 %b) {
    %tmp = xor i32 %a, %b
    %tmp1 = icmp ne i32 0, %tmp
    ret i1 %tmp1
}

define i1 @f4(i32 %a, i32 %b) {
    %tmp = xor i32 %a, %b
    %tmp1 = icmp eq i32 0, %tmp
    ret i1 %tmp1
}
