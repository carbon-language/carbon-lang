; RUN: opt -basicaa -loop-idiom < %s -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; CHECK: @memset
; CHECK-NOT: llvm.memset
define i8* @memset(i8* %b, i32 %c, i64 %len) nounwind uwtable ssp {
entry:
  %cmp1 = icmp ult i64 0, %len
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %conv6 = trunc i32 %c to i8
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvar = phi i64 [ 0, %for.body.lr.ph ], [ %indvar.next, %for.body ]
  %p.02 = getelementptr i8* %b, i64 %indvar
  store i8 %conv6, i8* %p.02, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar.next, %len
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret i8* %b
}

; CHECK: @memcpy
; CHECK-NOT: llvm.memcpy
define i8* @memcpy(i8* noalias %dst, i8* noalias %src, i64 %n) nounwind {
entry:
  %tobool3 = icmp eq i64 %n, 0
  br i1 %tobool3, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %c2.06 = phi i8* [ %incdec.ptr, %while.body ], [ %src, %entry ]
  %c1.05 = phi i8* [ %incdec.ptr1, %while.body ], [ %dst, %entry ]
  %n.addr.04 = phi i64 [ %dec, %while.body ], [ %n, %entry ]
  %dec = add i64 %n.addr.04, -1
  %incdec.ptr = getelementptr inbounds i8* %c2.06, i64 1
  %0 = load i8* %c2.06, align 1
  %incdec.ptr1 = getelementptr inbounds i8* %c1.05, i64 1
  store i8 %0, i8* %c1.05, align 1
  %tobool = icmp eq i64 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret i8* %dst
}

; CHECK: @memmove
; CHECK-NOT: llvm.memmove
define i8* @memmove(i8* %dst, i8* nocapture %src, i64 %count) nounwind {
entry:
  %sub = add i64 %count, -1
  %tobool9 = icmp eq i64 %count, 0
  br i1 %tobool9, label %while.end, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %entry
  %add.ptr2 = getelementptr inbounds i8* %src, i64 %sub
  %add.ptr = getelementptr inbounds i8* %dst, i64 %sub
  br label %while.body

while.body:                                       ; preds = %while.body.lr.ph, %while.body
  %b.012 = phi i8* [ %add.ptr2, %while.body.lr.ph ], [ %incdec.ptr, %while.body ]
  %a.011 = phi i8* [ %add.ptr, %while.body.lr.ph ], [ %incdec.ptr3, %while.body ]
  %count.addr.010 = phi i64 [ %count, %while.body.lr.ph ], [ %dec, %while.body ]
  %dec = add i64 %count.addr.010, -1
  %incdec.ptr = getelementptr inbounds i8* %b.012, i64 -1
  %0 = load i8* %b.012, align 1
  %incdec.ptr3 = getelementptr inbounds i8* %a.011, i64 -1
  store i8 %0, i8* %a.011, align 1
  %tobool = icmp eq i64 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret i8* %dst
}
