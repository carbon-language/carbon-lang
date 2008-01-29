; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: llvm-as -o - %s | llc -march=cellspu -mattr=large_mem > %t2.s
; RUN: grep shufb %t1.s | count 27
; RUN: grep   lqa %t1.s | count 27
; RUN: grep   lqd %t2.s | count 27
; RUN: grep space %t1.s | count 8
; RUN: grep  byte %t1.s | count 424
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define i32 @i32_extract_0(<4 x i32> %v) {
entry:
  %a = extractelement <4 x i32> %v, i32 0
  ret i32 %a
}

define i32 @i32_extract_1(<4 x i32> %v) {
entry:
  %a = extractelement <4 x i32> %v, i32 1
  ret i32 %a
}

define i32 @i32_extract_2(<4 x i32> %v) {
entry:
  %a = extractelement <4 x i32> %v, i32 2
  ret i32 %a
}

define i32 @i32_extract_3(<4 x i32> %v) {
entry:
  %a = extractelement <4 x i32> %v, i32 3
  ret i32 %a
}

define i16 @i16_extract_0(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 0
  ret i16 %a
}

define i16 @i16_extract_1(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 1
  ret i16 %a
}

define i16 @i16_extract_2(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 2
  ret i16 %a
}

define i16 @i16_extract_3(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 3
  ret i16 %a
}

define i16 @i16_extract_4(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 4
  ret i16 %a
}

define i16 @i16_extract_5(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 5
  ret i16 %a
}

define i16 @i16_extract_6(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 6
  ret i16 %a
}

define i16 @i16_extract_7(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 7
  ret i16 %a
}

define i8 @i8_extract_0(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 0
  ret i8 %a
}

define i8 @i8_extract_1(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 1
  ret i8 %a
}

define i8 @i8_extract_2(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 2
  ret i8 %a
}

define i8 @i8_extract_3(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 3
  ret i8 %a
}

define i8 @i8_extract_4(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 4
  ret i8 %a
}

define i8 @i8_extract_5(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 5
  ret i8 %a
}

define i8 @i8_extract_6(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 6
  ret i8 %a
}

define i8 @i8_extract_7(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 7
  ret i8 %a
}

define i8 @i8_extract_8(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 8
  ret i8 %a
}

define i8 @i8_extract_9(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 9
  ret i8 %a
}

define i8 @i8_extract_10(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 10
  ret i8 %a
}

define i8 @i8_extract_11(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 11
  ret i8 %a
}

define i8 @i8_extract_12(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 12
  ret i8 %a
}

define i8 @i8_extract_13(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 13
  ret i8 %a
}

define i8 @i8_extract_14(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 14
  ret i8 %a
}

define i8 @i8_extract_15(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 15
  ret i8 %a
}
