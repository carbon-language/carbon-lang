; RUN: not opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S  2>&1

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo() {
entry:
  ret i32 0
}
