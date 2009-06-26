; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {rsb\\W*r\[0-9\],\\W*r\[0-9\],\\W*#0} | count 1

define i32 @f1(i32 %a) {
    %tmp = sub i32 0, %a
    ret i32 %tmp
}
