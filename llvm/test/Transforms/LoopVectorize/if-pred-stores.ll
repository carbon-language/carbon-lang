; RUN: opt -S -vectorize-num-stores-pred=1 -force-vector-width=1 -force-vector-unroll=2 -loop-vectorize < %s | FileCheck %s --check-prefix=UNROLL
; RUN: opt -S -vectorize-num-stores-pred=1 -force-vector-width=2 -force-vector-unroll=1 -loop-vectorize -enable-cond-stores-vec < %s | FileCheck %s --check-prefix=VEC
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Test predication of stores.
define i32 @test(i32* nocapture %f) #0 {
entry:
  br label %for.body

; VEC-LABEL: test
; VEC:   %[[v8:.+]] = icmp sgt <2 x i32> %{{.*}}, <i32 100, i32 100>
; VEC:   %[[v9:.+]] = add nsw <2 x i32> %{{.*}}, <i32 20, i32 20>
; VEC:   %[[v10:.+]] = and <2 x i1> %[[v8]], <i1 true, i1 true>
; VEC:   %[[v11:.+]] = extractelement <2 x i1> %[[v10]], i32 0
; VEC:   %[[v12:.+]] = icmp eq i1 %[[v11]], true
; VEC:   br i1 %[[v12]], label %[[cond:.+]], label %[[else:.+]]
;
; VEC: [[cond]]:
; VEC:   %[[v13:.+]] = extractelement <2 x i32> %[[v9]], i32 0
; VEC:   %[[v14:.+]] = extractelement <2 x i32*> %{{.*}}, i32 0
; VEC:   store i32 %[[v13]], i32* %[[v14]], align 4
; VEC:   br label %[[else:.+]]
;
; VEC: [[else]]:
; VEC:   %[[v15:.+]] = extractelement <2 x i1> %[[v10]], i32 1
; VEC:   %[[v16:.+]] = icmp eq i1 %[[v15]], true
; VEC:   br i1 %[[v16]], label %[[cond2:.+]], label %[[else2:.+]]
;
; VEC: [[cond2]]:
; VEC:   %[[v17:.+]] = extractelement <2 x i32> %[[v9]], i32 1
; VEC:   %[[v18:.+]] = extractelement <2 x i32*> %{{.+}} i32 1
; VEC:   store i32 %[[v17]], i32* %[[v18]], align 4
; VEC:   br label %[[else2:.+]]
;
; VEC: [[else2]]:

; UNROLL-LABEL: test
; UNROLL: vector.body:
; UNROLL:   %[[IND:[a-zA-Z0-9]+]] = add i64 %{{.*}}, 0
; UNROLL:   %[[IND1:[a-zA-Z0-9]+]] = add i64 %{{.*}}, 1
; UNROLL:   %[[v0:[a-zA-Z0-9]+]] = getelementptr inbounds i32* %f, i64 %[[IND]]
; UNROLL:   %[[v1:[a-zA-Z0-9]+]] = getelementptr inbounds i32* %f, i64 %[[IND1]]
; UNROLL:   %[[v2:[a-zA-Z0-9]+]] = load i32* %[[v0]], align 4
; UNROLL:   %[[v3:[a-zA-Z0-9]+]] = load i32* %[[v1]], align 4
; UNROLL:   %[[v4:[a-zA-Z0-9]+]] = icmp sgt i32 %[[v2]], 100
; UNROLL:   %[[v5:[a-zA-Z0-9]+]] = icmp sgt i32 %[[v3]], 100
; UNROLL:   %[[v6:[a-zA-Z0-9]+]] = add nsw i32 %[[v2]], 20
; UNROLL:   %[[v7:[a-zA-Z0-9]+]] = add nsw i32 %[[v3]], 20
; UNROLL:   %[[v8:[a-zA-Z0-9]+]] = icmp eq i1 %[[v4]], true
; UNROLL:   br i1 %[[v8]], label %[[cond:[a-zA-Z0-9.]+]], label %[[else:[a-zA-Z0-9.]+]]
;
; UNROLL: [[cond]]:
; UNROLL:   store i32 %[[v6]], i32* %[[v0]], align 4
; UNROLL:   br label %[[else]]
;
; UNROLL: [[else]]:
; UNROLL:   %[[v9:[a-zA-Z0-9]+]] = icmp eq i1 %[[v5]], true
; UNROLL:   br i1 %[[v9]], label %[[cond2:[a-zA-Z0-9.]+]], label %[[else2:[a-zA-Z0-9.]+]]
;
; UNROLL: [[cond2]]:
; UNROLL:   store i32 %[[v7]], i32* %[[v1]], align 4
; UNROLL:   br label %[[else2]]
;
; UNROLL: [[else2]]:

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds i32* %f, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 100
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %add = add nsw i32 %0, 20
  store i32 %add, i32* %arrayidx, align 4
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 128
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 0
}
