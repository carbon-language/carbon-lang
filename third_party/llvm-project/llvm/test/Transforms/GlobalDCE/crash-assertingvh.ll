; Make sure that if a pass like jump threading populates a function analysis
; like LVI with asserting handles into the body of a function, those don't begin
; to assert when global DCE deletes the body of the function.
;
; RUN: opt -disable-output < %s -passes='module(function(jump-threading),globaldce)'
; RUN: opt -disable-output < %s -passes='module(rpo-function-attrs,globaldce)'

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @bar()

define internal i32 @foo() {
entry:
  %call4 = call i32 @bar()
  %cmp5 = icmp eq i32 %call4, 0
  br i1 %cmp5, label %if.then6, label %if.end8

if.then6:
  ret i32 0

if.end8:
  ret i32 1
}
