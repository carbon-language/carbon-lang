; RUN: opt -loop-vectorize -force-vector-width=2 -S %s | FileCheck %s

; Test cases to make sure LV & loop versioning can handle loops with
; multiple exiting branches.

; Multiple branches exiting the loop to a unique exit block. The loop should
; be vectorized with versioning & noalias metadata should be added.
define void @multiple_exits_unique_exit_block(i32* %A, i32* %B, i64 %N) {
; CHECK-LABEL: @multiple_exits_unique_exit_block
; CHECK:       vector.memcheck:
; CHECK-LABEL: vector.body:
; CHECK:         %wide.load = load <2 x i32>, <2 x i32>* {{.*}}, align 4, !alias.scope
; CHECK:         store <2 x i32> %wide.load, <2 x i32>* {{.*}}, align 4, !alias.scope
; CHECK:         br
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %cond.0 = icmp eq i64 %iv, %N
  br i1 %cond.0, label %exit, label %for.body

for.body:
  %A.gep = getelementptr inbounds i32, i32* %A, i64 %iv
  %lv = load i32, i32* %A.gep, align 4
  %B.gep = getelementptr inbounds i32, i32* %B, i64 %iv
  store i32 %lv, i32* %B.gep, align 4
  %iv.next = add nuw i64 %iv, 1
  %cond.1 = icmp ult i64 %iv.next, 1000
  br i1 %cond.1, label %loop.header, label %exit

exit:
  ret void
}


; Multiple branches exiting the loop to different blocks. Currently this is not supported.
define i32 @multiple_exits_multiple_exit_blocks(i32* %A, i32* %B, i64 %N) {
; CHECK-LABEL: @multiple_exits_multiple_exit_blocks
; CHECK-NEXT:    entry:
; CHECK:           br label %loop.header
; CHECK-NOT:      <2 x i32>
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %cond.0 = icmp eq i64 %iv, %N
  br i1 %cond.0, label %exit.0, label %for.body

for.body:
  %A.gep = getelementptr inbounds i32, i32* %A, i64 %iv
  %lv = load i32, i32* %A.gep, align 4
  %B.gep = getelementptr inbounds i32, i32* %B, i64 %iv
  store i32 %lv, i32* %B.gep, align 4
  %iv.next = add nuw i64 %iv, 1
  %cond.1 = icmp ult i64 %iv.next, 1000
  br i1 %cond.1, label %loop.header, label %exit.1

exit.0:
  ret i32 1

exit.1:
  ret i32 2
}
