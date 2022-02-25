; RUN: opt -newgvn -S < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @print_long_format()
define dso_local void @print_long_format() #0 {
entry:
  switch i32 undef, label %sw.default [
    i32 1, label %sw.bb
    i32 0, label %sw.bb19
    i32 2, label %sw.bb23
  ]

sw.bb:                                            ; preds = %entry
  unreachable

sw.bb19:                                          ; preds = %entry
  br i1 undef, label %if.then37, label %if.end50

sw.bb23:                                          ; preds = %entry
  unreachable

sw.default:                                       ; preds = %entry
  unreachable

if.then37:                                        ; preds = %sw.bb19
  unreachable

if.end50:                                         ; preds = %sw.bb19
  %call180 = call i32 @timespec_cmp() #2
  %cmp181 = icmp slt i32 %call180, 0
  ret void
}

; Function Attrs: writeonly
declare dso_local i32 @timespec_cmp() #1

attributes #0 = { "use-soft-float"="false" }
attributes #1 = { writeonly }
attributes #2 = { nounwind readonly }

