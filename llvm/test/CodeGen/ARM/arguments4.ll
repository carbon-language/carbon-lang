; RUN: llvm-as < %s | llc -mtriple=arm-linux-gnueabi
; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin

define float @f(i32 %a, i128 %b) {
        %tmp = call float @g(i128 %b)
        ret float %tmp
}

declare float @g(i128)
