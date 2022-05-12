; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; Formerly there were two shifts.

define i64 @baz(i32 %A) nounwind {
; CHECK:  shlq  $49, %r
        %tmp1 = shl i32 %A, 17
        %tmp2 = zext i32 %tmp1 to i64
        %tmp3 = shl i64 %tmp2, 32
        ret i64 %tmp3
}
