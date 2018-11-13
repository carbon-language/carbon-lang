; RUN: opt -S -loop-simplify < %s | FileCheck %s

; Two-loop nest with llvm.loop metadata on each loop.
; inner.header exits to outer.header. inner.header is a latch for the outer
; loop, and contains the outer loop's metadata.
; After loop-simplify, a new block "outer.header.loopexit" is created between
; inner.header and outer.header. The metadata from inner.header must be moved
; to the new block, as the new block becomes the outer loop latch.
; The metadata on the inner loop's latch should be untouched.

; CHECK: outer.header.loopexit:
; CHECK-NEXT: llvm.loop [[UJAMTAG:.*]]
; CHECK-NOT: br i1 {{.*}}, label {{.*}}, label %outer.header.loopexit, !llvm.loop
; CHECK: br label %inner.header, !llvm.loop [[UNROLLTAG:.*]]

; CHECK: distinct !{[[UJAMTAG]], [[UJAM:.*]]}
; CHECK: [[UJAM]] = !{!"llvm.loop.unroll_and_jam.count", i32 17}
; CHECK: distinct !{[[UNROLLTAG]], [[UNROLL:.*]]}
; CHECK: [[UNROLL]] = !{!"llvm.loop.unroll.count", i32 1}


define dso_local void @loopnest() local_unnamed_addr #0 {
entry:
  br label %outer.header

outer.header:                                         ; preds = %inner.header, %entry
  %ii.0 = phi i64 [ 2, %entry ], [ %add, %inner.header ]
  %cmp = icmp ult i64 %ii.0, 64
  br i1 %cmp, label %inner.header, label %outer.header.cleanup

outer.header.cleanup:                                 ; preds = %outer.header
  ret void

inner.header:                                        ; preds = %outer.header, %inner.body
  %j.0 = phi i64 [ %add10, %inner.body ], [ %ii.0, %outer.header ]
  %add = add nuw nsw i64 %ii.0, 16
  %cmp2 = icmp ult i64 %j.0, %add
  br i1 %cmp2, label %inner.body, label %outer.header, !llvm.loop !2

inner.body:                                        ; preds = %inner.header
  %add10 = add nuw nsw i64 %j.0, 1
  br label %inner.header, !llvm.loop !4
}

!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.unroll_and_jam.count", i32 17}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.unroll.count", i32 1}
