; RUN: llc < %s -mtriple=arm-linux-gnueabi | \
; RUN:   grep {mov r0, r2} | count 1
; RUN: llc < %s -mtriple=arm-apple-darwin | \
; RUN:   grep {mov r0, r1} | count 1

define i32 @f(i32 %a, i64 %b) {
        %tmp = call i32 @g(i64 %b)
        ret i32 %tmp
}

declare i32 @g(i64)
