; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+sse41,-avx < %s | FileCheck %s --check-prefix SSE41
; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+avx < %s | FileCheck %s --check-prefix AVX

define i32 @veccond(<4 x i32> %input) {
entry:
  %0 = bitcast <4 x i32> %input to i128
  %1 = icmp ne i128 %0, 0
  br i1 %1, label %if-true-block, label %endif-block

if-true-block:                                    ; preds = %entry
  ret i32 0
endif-block:                                      ; preds = %entry,
  ret i32 1
; SSE41: veccond
; SSE41: ptest
; SSE41: ret
; AVX:   veccond
; AVX:   vptest
; AVX:   ret
}

define i32 @vectest(<4 x i32> %input) {
entry:
  %0 = bitcast <4 x i32> %input to i128
  %1 = icmp ne i128 %0, 0
  %2 = zext i1 %1 to i32
  ret i32 %2
; SSE41: vectest
; SSE41: ptest
; SSE41: ret
; AVX:   vectest
; AVX:   vptest
; AVX:   ret
}

define i32 @vecsel(<4 x i32> %input, i32 %a, i32 %b) {
entry:
  %0 = bitcast <4 x i32> %input to i128
  %1 = icmp ne i128 %0, 0
  %2 = select i1 %1, i32 %a, i32 %b
  ret i32 %2
; SSE41: vecsel
; SSE41: ptest
; SSE41: ret
; AVX:   vecsel
; AVX:   vptest
; AVX:   ret
}
