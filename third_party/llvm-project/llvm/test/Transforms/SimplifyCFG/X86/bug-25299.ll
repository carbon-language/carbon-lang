; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

;; Test case for bug 25299, contributed by David Majnemer.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i1 %B) personality i1 undef {
entry:
;CHECK: entry
;CHECK-NEXT: call void @g()
  invoke void @g()
          to label %continue unwind label %unwind

unwind:                                           ; preds = %entry
  %tmp101 = landingpad { i8*, i32 }
          cleanup
  br i1 %B, label %resume, label %then

then:                                             ; preds = %cleanup1
  br label %resume

resume:                                           ; preds = %cleanup2, %then, %cleanup1, %unwind
  %tmp104 = phi { i8*, i32 } [ %tmp101, %then ], [ %tmp106, %cleanup2 ], [ %tmp101, %unwind ]
;CHECK-NOT: resume { i8*, i32 } %tmp104
  resume { i8*, i32 } %tmp104

continue:                                         ; preds = %entry, %continue
;CHECK: continue:                                         ; preds = %entry, %continue
;CHECK-NEXT: call void @g()
  invoke void @g()
          to label %continue unwind label %cleanup2

cleanup2:                                         ; preds = %continue
  %tmp106 = landingpad { i8*, i32 }
          cleanup
  br label %resume
}

declare void @g()