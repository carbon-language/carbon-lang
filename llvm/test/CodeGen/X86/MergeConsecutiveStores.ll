; RUN: llc -march=x86-64 -mcpu=corei7 < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct.A = type { i8, i8, i8, i8, i8, i8, i8, i8 }

@a = common global [10000 x %struct.A] zeroinitializer, align 8

; Move all of the constants using a single vector store.
; CHECK: merge_const_store
; CHECK: movq %xmm0
; CHECK: ret
define void @merge_const_store(i32 %count, %struct.A* nocapture %p) nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge
.lr.ph:
  %i.02 = phi i32 [ %10, %.lr.ph ], [ 0, %0 ]
  %.01 = phi %struct.A* [ %11, %.lr.ph ], [ %p, %0 ]
  %2 = getelementptr inbounds %struct.A* %.01, i64 0, i32 0
  store i8 1, i8* %2, align 1
  %3 = getelementptr inbounds %struct.A* %.01, i64 0, i32 1
  store i8 2, i8* %3, align 1
  %4 = getelementptr inbounds %struct.A* %.01, i64 0, i32 2
  store i8 3, i8* %4, align 1
  %5 = getelementptr inbounds %struct.A* %.01, i64 0, i32 3
  store i8 4, i8* %5, align 1
  %6 = getelementptr inbounds %struct.A* %.01, i64 0, i32 4
  store i8 5, i8* %6, align 1
  %7 = getelementptr inbounds %struct.A* %.01, i64 0, i32 5
  store i8 6, i8* %7, align 1
  %8 = getelementptr inbounds %struct.A* %.01, i64 0, i32 6
  store i8 7, i8* %8, align 1
  %9 = getelementptr inbounds %struct.A* %.01, i64 0, i32 7
  store i8 8, i8* %9, align 1
  %10 = add nsw i32 %i.02, 1
  %11 = getelementptr inbounds %struct.A* %.01, i64 1
  %exitcond = icmp eq i32 %10, %count
  br i1 %exitcond, label %._crit_edge, label %.lr.ph
._crit_edge:
  ret void
}

; Move the first 4 constants as a single vector. Move the rest as scalars.
; CHECK: merge_nonconst_store
; CHECK: movd %xmm0
; CHECK: movb
; CHECK: movb
; CHECK: movb
; CHECK: movb
; CHECK: ret
define void @merge_nonconst_store(i32 %count, i8 %zz, %struct.A* nocapture %p) nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge
.lr.ph:
  %i.02 = phi i32 [ %10, %.lr.ph ], [ 0, %0 ]
  %.01 = phi %struct.A* [ %11, %.lr.ph ], [ %p, %0 ]
  %2 = getelementptr inbounds %struct.A* %.01, i64 0, i32 0
  store i8 1, i8* %2, align 1
  %3 = getelementptr inbounds %struct.A* %.01, i64 0, i32 1
  store i8 2, i8* %3, align 1
  %4 = getelementptr inbounds %struct.A* %.01, i64 0, i32 2
  store i8 3, i8* %4, align 1
  %5 = getelementptr inbounds %struct.A* %.01, i64 0, i32 3
  store i8 4, i8* %5, align 1
  %6 = getelementptr inbounds %struct.A* %.01, i64 0, i32 4
  store i8 %zz, i8* %6, align 1                     ;  <----------- Not a const;
  %7 = getelementptr inbounds %struct.A* %.01, i64 0, i32 5
  store i8 6, i8* %7, align 1
  %8 = getelementptr inbounds %struct.A* %.01, i64 0, i32 6
  store i8 7, i8* %8, align 1
  %9 = getelementptr inbounds %struct.A* %.01, i64 0, i32 7
  store i8 8, i8* %9, align 1
  %10 = add nsw i32 %i.02, 1
  %11 = getelementptr inbounds %struct.A* %.01, i64 1
  %exitcond = icmp eq i32 %10, %count
  br i1 %exitcond, label %._crit_edge, label %.lr.ph
._crit_edge:
  ret void
}


;CHECK: merge_loads
; load:
;CHECK: movw
; store:
;CHECK: movw
;CHECK: ret
define void @merge_loads(i32 %count, %struct.A* noalias nocapture %q, %struct.A* noalias nocapture %p) nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = getelementptr inbounds %struct.A* %q, i64 0, i32 0
  %3 = getelementptr inbounds %struct.A* %q, i64 0, i32 1
  br label %4

; <label>:4                                       ; preds = %4, %.lr.ph
  %i.02 = phi i32 [ 0, %.lr.ph ], [ %9, %4 ]
  %.01 = phi %struct.A* [ %p, %.lr.ph ], [ %10, %4 ]
  %5 = load i8* %2, align 1
  %6 = load i8* %3, align 1
  %7 = getelementptr inbounds %struct.A* %.01, i64 0, i32 0
  store i8 %5, i8* %7, align 1
  %8 = getelementptr inbounds %struct.A* %.01, i64 0, i32 1
  store i8 %6, i8* %8, align 1
  %9 = add nsw i32 %i.02, 1
  %10 = getelementptr inbounds %struct.A* %.01, i64 1
  %exitcond = icmp eq i32 %9, %count
  br i1 %exitcond, label %._crit_edge, label %4

._crit_edge:                                      ; preds = %4, %0
  ret void
}

; The loads and the stores are interleved. Can't merge them.
;CHECK: no_merge_loads
;CHECK: movb
;CHECK: movb
;CHECK: movb
;CHECK: movb
;CHECK: ret
define void @no_merge_loads(i32 %count, %struct.A* noalias nocapture %q, %struct.A* noalias nocapture %p) nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = getelementptr inbounds %struct.A* %q, i64 0, i32 0
  %3 = getelementptr inbounds %struct.A* %q, i64 0, i32 1
  br label %a4

a4:                                       ; preds = %4, %.lr.ph
  %i.02 = phi i32 [ 0, %.lr.ph ], [ %a9, %a4 ]
  %.01 = phi %struct.A* [ %p, %.lr.ph ], [ %a10, %a4 ]
  %a5 = load i8* %2, align 1
  %a7 = getelementptr inbounds %struct.A* %.01, i64 0, i32 0
  store i8 %a5, i8* %a7, align 1
  %a8 = getelementptr inbounds %struct.A* %.01, i64 0, i32 1
  %a6 = load i8* %3, align 1
  store i8 %a6, i8* %a8, align 1
  %a9 = add nsw i32 %i.02, 1
  %a10 = getelementptr inbounds %struct.A* %.01, i64 1
  %exitcond = icmp eq i32 %a9, %count
  br i1 %exitcond, label %._crit_edge, label %a4

._crit_edge:                                      ; preds = %4, %0
  ret void
}


