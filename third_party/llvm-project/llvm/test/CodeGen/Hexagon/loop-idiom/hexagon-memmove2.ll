; RUN: opt -basic-aa -hexagon-loop-idiom -S -mtriple hexagon-unknown-elf < %s \
; RUN:  | FileCheck %s

define void @PR14241(i32* %s, i64 %size) #0 {
; Ensure that we don't form a memcpy for strided loops. Briefly, when we taught
; LoopIdiom about memmove and strided loops, this got miscompiled into a memcpy
; instead of a memmove. If we get the memmove transform back, this will catch
; regressions.
;
; CHECK-LABEL: @PR14241(

entry:
  %end.idx = add i64 %size, -1
  %end.ptr = getelementptr inbounds i32, i32* %s, i64 %end.idx
  br label %while.body
; CHECK-NOT: memcpy
; CHECK: memmove

while.body:
  %phi.ptr = phi i32* [ %s, %entry ], [ %next.ptr, %while.body ]
  %src.ptr = getelementptr inbounds i32, i32* %phi.ptr, i64 1
  %val = load i32, i32* %src.ptr, align 4
; CHECK: load
  %dst.ptr = getelementptr inbounds i32, i32* %phi.ptr, i64 0
  store i32 %val, i32* %dst.ptr, align 4
; CHECK: store
  %next.ptr = getelementptr inbounds i32, i32* %phi.ptr, i64 1
  %cmp = icmp eq i32* %next.ptr, %end.ptr
  br i1 %cmp, label %exit, label %while.body

exit:
  ret void
; CHECK: ret void
}

attributes #0 = { nounwind }
