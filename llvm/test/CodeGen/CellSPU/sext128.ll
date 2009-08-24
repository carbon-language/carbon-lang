; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep {long.*269488144} %t1.s | count 2
; RUN: grep {long.*16909060}  %t1.s | count 1
; RUN: grep {long.*84281096}  %t1.s | count 1
; RUN: grep {rotmai}          %t1.s | count 1
; RUN: grep {lqa}             %t1.s | count 1
; RUN: grep {shufb}           %t1.s | count 1

; ModuleID = 'sext128.bc'
target datalayout = "E-p:32:32:128-i1:8:128-i8:8:128-i16:16:128-i32:32:128-i64:32:128-f32:32:128-f64:64:128-v64:128:128-v128:128:128-a0:0:128-s0:128:128"
target triple = "spu"

define i128 @sext_i64_i128(i64 %a) {
entry:
        %0 = sext i64 %a to i128
        ret i128 %0
}
