; RUN: llc < %s -march=x86 -mcpu=corei7 -mtriple=i686-pc-win32 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; Make sure that we are zeroing one memory location at a time using xorl and
; not both using XMM registers.

;CHECK: @foo
;CHECK: xorl
;CHECK-NOT: xmm
;CHECK: ret
define i32 @foo (i64* %so) nounwind uwtable ssp {
entry:
  %used = getelementptr inbounds i64* %so, i32 3
  store i64 0, i64* %used, align 8
  %fill = getelementptr inbounds i64* %so, i32 2
  %L = load i64* %fill, align 8
  store i64 0, i64* %fill, align 8
  %cmp28 = icmp sgt i64 %L, 0
  %R = sext i1 %cmp28 to i32
  ret i32 %R
}
