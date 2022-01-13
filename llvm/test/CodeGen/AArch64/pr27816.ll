; RUN: llc %s -mtriple=aarch64 -o - | FileCheck %s

%struct.A = type { i8, i8, i8, i8, i8, i8, i8, i8, i32 }

; The existence of the final i32 value should not prevent the i8s from
; being merged.

; CHECK-LABEL: @merge_const_store
; CHECK-NOT: strb
; CHECK: str x8,  [x1]
; CHECK-NOT: strb
; CHECK: str wzr, [x1, #8]
; CHECK-NOT: strb
define void @merge_const_store(i32 %count, %struct.A* nocapture %p)  {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge
.lr.ph:
  %i.02 = phi i32 [ %add, %.lr.ph ], [ 0, %0 ]
  %.01 = phi %struct.A* [ %addr, %.lr.ph ], [ %p, %0 ]
  %a2 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 0
  store i8 1, i8* %a2, align 1
  %a3 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 1
  store i8 2, i8* %a3, align 1
  %a4 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 2
  store i8 3, i8* %a4, align 1
  %a5 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 3
  store i8 4, i8* %a5, align 1
  %a6 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 4
  store i8 5, i8* %a6, align 1
  %a7 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 5
  store i8 6, i8* %a7, align 1
  %a8 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 6
  store i8 7, i8* %a8, align 1
  %a9 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 7
  store i8 8, i8* %a9, align 1

  ;
  %addr_last = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 8
  store i32 0, i32* %addr_last, align 4


  %add = add nsw i32 %i.02, 1
  %addr = getelementptr inbounds %struct.A, %struct.A* %.01, i64 1
  %exitcond = icmp eq i32 %add, %count
  br i1 %exitcond, label %._crit_edge, label %.lr.ph
._crit_edge:
  ret void
}
