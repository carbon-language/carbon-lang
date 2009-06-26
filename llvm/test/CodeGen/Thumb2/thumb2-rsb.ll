; XFAIL: *
; this will match as "sub" until we get register shifting

; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {rsb\\W*r\[0-9\],\\W*r\[0-9\],\\W*r\[0-9\]*} | count 1

define i32 @f1(i32 %a, i32 %b) {
    %tmp = sub i32 %b, %a
    ret i32 %tmp
}
