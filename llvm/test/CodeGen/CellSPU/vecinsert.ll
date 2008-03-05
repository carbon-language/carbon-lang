; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep cbd     %t1.s | count 3 
; RUN: grep chd     %t1.s | count 3
; RUN: grep cwd     %t1.s | count 6
; RUN: grep il      %t1.s | count 4
; RUN: grep ilh     %t1.s | count 3
; RUN: grep iohl    %t1.s | count 1
; RUN: grep ilhu    %t1.s | count 1
; RUN: grep shufb   %t1.s | count 12
; RUN: grep 17219   %t1.s | count 1 
; RUN: grep 22598   %t1.s | count 1
; RUN: grep -- -39  %t1.s | count 1
; RUN: grep    24   %t1.s | count 1
; RUN: grep  1159   %t1.s | count 1
; ModuleID = 'vecinsert.bc'
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128"
target triple = "spu-unknown-elf"

; 67 -> 0x43, as 8-bit vector constant load = 0x4343 (17219)0x4343
define <16 x i8> @test_v16i8(<16 x i8> %P, i8 %x) {
entry:
        %tmp1 = insertelement <16 x i8> %P, i8 %x, i32 10
        %tmp1.1 = insertelement <16 x i8> %tmp1, i8 67, i32 7
        %tmp1.2 = insertelement <16 x i8> %tmp1.1, i8 %x, i32 15
        ret <16 x i8> %tmp1.2
}

; 22598 -> 0x5846
define <8 x i16> @test_v8i16(<8 x i16> %P, i16 %x) {
entry:
        %tmp1 = insertelement <8 x i16> %P, i16 %x, i32 5
        %tmp1.1 = insertelement <8 x i16> %tmp1, i16 22598, i32 7
        %tmp1.2 = insertelement <8 x i16> %tmp1.1, i16 %x, i32 2
        ret <8 x i16> %tmp1.2
}

; 1574023 -> 0x180487 (ILHU 24/IOHL 1159)
define <4 x i32> @test_v4i32_1(<4 x i32> %P, i32 %x) {
entry:
        %tmp1 = insertelement <4 x i32> %P, i32 %x, i32 2
        %tmp1.1 = insertelement <4 x i32> %tmp1, i32 1574023, i32 1
        %tmp1.2 = insertelement <4 x i32> %tmp1.1, i32 %x, i32 3
        ret <4 x i32> %tmp1.2
}

; Should generate IL for the load
define <4 x i32> @test_v4i32_2(<4 x i32> %P, i32 %x) {
entry:
        %tmp1 = insertelement <4 x i32> %P, i32 %x, i32 2
        %tmp1.1 = insertelement <4 x i32> %tmp1, i32 -39, i32 1
        %tmp1.2 = insertelement <4 x i32> %tmp1.1, i32 %x, i32 3
        ret <4 x i32> %tmp1.2
}
