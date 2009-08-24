; RUN: llvm-as -o - %s | llc -march=cellspu | FileCheck %s 

; ModuleID = 'sext128.bc'
target datalayout = "E-p:32:32:128-i1:8:128-i8:8:128-i16:16:128-i32:32:128-i64:32:128-f32:32:128-f64:64:128-v64:128:128-v128:128:128-a0:0:128-s0:128:128"
target triple = "spu"

define i128 @sext_i64_i128(i64 %a) {
entry:
; CHECK: 	long	269488144
; CHECK: 	long	269488144
; CHECK:	long	16909060
; CHECK: 	long	84281096
; CHECK: 	rotmai
; CHECK:	lqa
; CHECK:	shufb
        %0 = sext i64 %a to i128
        ret i128 %0
}
