; RUN: opt %loadPolly -polly-print-scops -disable-output < %s -stats 2>&1 | FileCheck %s --check-prefix=RTA
; RUN: opt %loadPolly -polly-print-scops -polly-use-runtime-alias-checks=false -disable-output < %s -stats 2>&1 | FileCheck %s --check-prefix=NORTA
; REQUIRES: asserts

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @f(i32** nocapture %ptrs, i64 %p0, i64 %p1, i64 %p2) nounwind {
bb.nph:
  %0 = getelementptr inbounds i32*, i32** %ptrs, i64 %p0 ; <i32**> [#uses=1]
  %1 = load i32*, i32** %0, align 8                     ; <i32*> [#uses=1]
  %2 = getelementptr inbounds i32*, i32** %ptrs, i64 %p1 ; <i32**> [#uses=1]
  %3 = load i32*, i32** %2, align 8                     ; <i32*> [#uses=1]
  %4 = getelementptr inbounds i32*, i32** %ptrs, i64 %p2 ; <i32**> [#uses=1]
  %5 = load i32*, i32** %4, align 8                     ; <i32*> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %i.03 = phi i64 [ 0, %bb.nph ], [ %tmp, %bb ]   ; <i64> [#uses=3]
  %scevgep = getelementptr i32, i32* %3, i64 %i.03     ; <i32*> [#uses=1]
  %scevgep4 = getelementptr i32, i32* %5, i64 %i.03    ; <i32*> [#uses=1]
  %tmp = add i64 %i.03, 1                         ; <i64> [#uses=3]
  %scevgep5 = getelementptr i32, i32* %1, i64 %tmp     ; <i32*> [#uses=1]
  %6 = load i32, i32* %scevgep, align 4                ; <i32> [#uses=1]
  %7 = load i32, i32* %scevgep4, align 4               ; <i32> [#uses=1]
  %8 = add nsw i32 %7, %6                         ; <i32> [#uses=1]
  store i32 %8, i32* %scevgep5, align 4
  %exitcond = icmp eq i64 %tmp, 64                ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb
  ret void
}

; RTA:   1 polly-detect     - Number of scops
; NORTA: 1 polly-detect     - Number of rejected regions: Base address aliasing
