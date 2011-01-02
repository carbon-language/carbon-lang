; RUN: opt -basicaa -loop-idiom < %s -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

define void @test1(i8* %Base, i64 %Size) nounwind ssp {
bb.nph:                                           ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8* %Base, i64 %indvar
  store i8 0, i8* %I.0.014, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test1
; CHECK: call void @llvm.memset.p0i8.i64(i8* %Base, i8 0, i64 %Size, i32 1, i1 false)
; CHECK-NOT: store
}

; This is a loop that was rotated but where the blocks weren't merged.  This
; shouldn't perturb us.
define void @test1a(i8* %Base, i64 %Size) nounwind ssp {
bb.nph:                                           ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body.cont ]
  %I.0.014 = getelementptr i8* %Base, i64 %indvar
  store i8 0, i8* %I.0.014, align 1
  %indvar.next = add i64 %indvar, 1
  br label %for.body.cont
for.body.cont:
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test1a
; CHECK: call void @llvm.memset.p0i8.i64(i8* %Base, i8 0, i64 %Size, i32 1, i1 false)
; CHECK-NOT: store
}


define void @test2(i32* %Base, i64 %Size) nounwind ssp {
entry:
  %cmp10 = icmp eq i64 %Size, 0
  br i1 %cmp10, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.011 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %add.ptr.i = getelementptr i32* %Base, i64 %i.011
  store i32 16843009, i32* %add.ptr.i, align 4
  %inc = add nsw i64 %i.011, 1
  %exitcond = icmp eq i64 %inc, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test2
; CHECK: br i1 %cmp10,
; CHECK: %tmp = mul i64 %Size, 4
; CHECK: call void @llvm.memset.p0i8.i64(i8* %Base1, i8 1, i64 %tmp, i32 4, i1 false)
; CHECK-NOT: store
}

; This is a case where there is an extra may-aliased store in the loop, we can't
; promote the memset.
define void @test3(i32* %Base, i64 %Size, i8 *%MayAlias) nounwind ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.011 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %add.ptr.i = getelementptr i32* %Base, i64 %i.011
  store i32 16843009, i32* %add.ptr.i, align 4
  
  store i8 42, i8* %MayAlias
  %inc = add nsw i64 %i.011, 1
  %exitcond = icmp eq i64 %inc, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %entry
  ret void
; CHECK: @test3
; CHECK-NOT: memset
; CHECK: ret void
}


;; TODO: We should be able to promote this memset.  Not yet though.
define void @test4(i8* %Base) nounwind ssp {
bb.nph:                                           ; preds = %entry
  %Base100 = getelementptr i8* %Base, i64 1000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8* %Base, i64 %indvar
  store i8 0, i8* %I.0.014, align 1
  
  ;; Store beyond the range memset, should be safe to promote.
  store i8 42, i8* %Base100
  
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 100
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK-TODO: @test4
; CHECK-TODO: call void @llvm.memset.p0i8.i64(i8* %Base, i8 0, i64 100, i32 1, i1 false)
; CHECK-TODO-NOT: store
}

; This can't be promoted: the memset is a store of a loop variant value.
define void @test5(i8* %Base, i64 %Size) nounwind ssp {
bb.nph:                                           ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8* %Base, i64 %indvar
  
  %V = trunc i64 %indvar to i8
  store i8 %V, i8* %I.0.014, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test5
; CHECK-NOT: memset
; CHECK: ret void
}


;; memcpy formation
define void @test6(i64 %Size) nounwind ssp {
bb.nph:
  %Base = alloca i8, i32 10000
  %Dest = alloca i8, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8* %Base, i64 %indvar
  %DestI = getelementptr i8* %Dest, i64 %indvar
  %V = load i8* %I.0.014, align 1
  store i8 %V, i8* %DestI, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test6
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %Dest, i8* %Base, i64 %Size, i32 1, i1 false)
; CHECK-NOT: store
; CHECK: ret void
}


; This is a loop that was rotated but where the blocks weren't merged.  This
; shouldn't perturb us.
define void @test7(i8* %Base, i64 %Size) nounwind ssp {
bb.nph:                                           ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body.cont ]
  br label %for.body.cont
for.body.cont:
  %I.0.014 = getelementptr i8* %Base, i64 %indvar
  store i8 0, i8* %I.0.014, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test7
; CHECK: call void @llvm.memset.p0i8.i64(i8* %Base, i8 0, i64 %Size, i32 1, i1 false)
; CHECK-NOT: store
}


