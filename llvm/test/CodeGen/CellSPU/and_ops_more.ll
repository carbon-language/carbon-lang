; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep and    %t1.s | count 10
; RUN: not grep andc %t1.s
; RUN: not grep andi %t1.s
; RUN: grep andhi  %t1.s | count 5
; RUN: grep andbi  %t1.s | count 1
; XFAIL: *

; This testcase is derived from test/CodeGen/CellSPU/and_ops.ll and
; records the changes due to r50358. The and_sext8 function appears
; to be improved by this change, while the andhi_i16 function appears
; to be pessimized.

target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define i16 @andhi_i16(i16 signext  %in) signext  {
        %tmp38 = and i16 %in, 37         ; <i16> [#uses=1]
        ret i16 %tmp38
}

define i8 @and_sext8(i8 signext  %in) signext  {
        ; ANDBI generated
        %tmp38 = and i8 %in, 37
        ret i8 %tmp38
}
