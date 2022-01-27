; RUN: llc -mtriple=thumbv6-apple-darwin < %s | FileCheck %s
; r8869722

%struct.state = type { i32, %struct.info*, float**, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64, i64, i8* }
%struct.info = type { i32, i32, i32, i32, i32, i32, i32, i8* }

define void @t1(%struct.state* %v) {
; CHECK: push {r4
  %tmp6 = load i32, i32* null
  %tmp8 = alloca float, i32 %tmp6
  store i32 1, i32* null
  br label %return

return:                                           ; preds = %0
; CHECK: mov sp, r4
  ret void
}
