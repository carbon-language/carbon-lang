; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {cmp\\W*r\[0-9\],\\W*r\[0-9\]} | count 2

define i1 @f1(i32 %a, i32 %b) {
    %tmp = icmp ne i32 %a, %b
    ret i1 %tmp
}

define i1 @f2(i32 %a, i32 %b) {
    %tmp = icmp eq i32 %a, %b
    ret i1 %tmp
}
