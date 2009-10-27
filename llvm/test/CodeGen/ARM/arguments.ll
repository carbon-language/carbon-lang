; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s -check-prefix=ELF
; RUN: llc < %s -mtriple=arm-apple-darwin  | FileCheck %s -check-prefix=DARWIN

define i32 @f(i32 %a, i64 %b) {
; ELF: mov r0, r2
; DARWIN: mov r0, r1
        %tmp = call i32 @g(i64 %b)
        ret i32 %tmp
}

declare i32 @g(i64)
