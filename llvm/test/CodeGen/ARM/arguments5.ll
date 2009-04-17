; RUN: llvm-as < %s | llc -mtriple=arm-linux-gnueabi
; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin

define double @f(i32 %a, i128 %b) {
        %tmp = call double @g(i128 %b)
        ret double %tmp
}

declare double @g(i128)
