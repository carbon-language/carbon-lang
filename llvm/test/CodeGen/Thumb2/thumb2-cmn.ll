; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {cmn\\W*r\[0-9\],\\W*r\[0-9\]} | count 4

define i1 @f1(i32 %a, i32 %b) {
    %nb = sub i32 0, %b
    %tmp = icmp ne i32 %a, %nb
    ret i1 %tmp
}

define i1 @f2(i32 %a, i32 %b) {
    %nb = sub i32 0, %b
    %tmp = icmp ne i32 %nb, %a
    ret i1 %tmp
}

define i1 @f3(i32 %a, i32 %b) {
    %nb = sub i32 0, %b
    %tmp = icmp eq i32 %a, %nb
    ret i1 %tmp
}

define i1 @f4(i32 %a, i32 %b) {
    %nb = sub i32 0, %b
    %tmp = icmp eq i32 %nb, %a
    ret i1 %tmp
}
