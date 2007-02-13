; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -mtriple=arm-linux | grep "mov r0, r2" | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin | grep "mov r0, r1" | wc -l | grep 1

define i32 @f(i32 %a, i64 %b) {
        %tmp = call i32 @g(i64 %b)
        ret i32 %tmp
}

declare i32 @g(i64)