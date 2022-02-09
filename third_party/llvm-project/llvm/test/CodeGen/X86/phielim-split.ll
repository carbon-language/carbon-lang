; RUN: llc < %s -verify-machineinstrs -no-phi-elim-live-out-early-exit | FileCheck %s
target triple = "x86_64-apple-macosx10.8.0"

; The critical edge from for.cond to if.end2 should be split to avoid injecting
; copies into the loop. The use of %b after the loop causes interference that
; makes a copy necessary.
; <rdar://problem/11561842>
;
; CHECK: split_loop_exit
; CHECK: %for.cond
; CHECK-NOT: mov
; CHECK: je

define i32 @split_loop_exit(i32 %a, i32 %b, i8* nocapture %p) nounwind uwtable readonly ssp {
entry:
  %cmp = icmp sgt i32 %a, 10
  br i1 %cmp, label %for.cond, label %if.end2

for.cond:                                         ; preds = %entry, %for.cond
  %p.addr.0 = phi i8* [ %incdec.ptr, %for.cond ], [ %p, %entry ]
  %incdec.ptr = getelementptr inbounds i8, i8* %p.addr.0, i64 1
  %0 = load i8, i8* %p.addr.0, align 1
  %tobool = icmp eq i8 %0, 0
  br i1 %tobool, label %for.cond, label %if.end2

if.end2:                                          ; preds = %for.cond, %entry
  %r.0 = phi i32 [ %a, %entry ], [ %b, %for.cond ]
  %add = add nsw i32 %r.0, %b
  ret i32 %add
}

; CHECK: split_live_out
; CHECK: %while.body
; CHECK: cmp
; CHECK-NEXT: ja
define i8* @split_live_out(i32 %value, i8* %target) nounwind uwtable readonly ssp {
entry:
  %cmp10 = icmp ugt i32 %value, 127
  br i1 %cmp10, label %while.body.preheader, label %while.end

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %target.addr.012 = phi i8* [ %incdec.ptr, %while.body ], [ %target, %while.body.preheader ]
  %value.addr.011 = phi i32 [ %shr, %while.body ], [ %value, %while.body.preheader ]
  %or = or i32 %value.addr.011, 128
  %conv = trunc i32 %or to i8
  store i8 %conv, i8* %target.addr.012, align 1
  %shr = lshr i32 %value.addr.011, 7
  %incdec.ptr = getelementptr inbounds i8, i8* %target.addr.012, i64 1
  %cmp = icmp ugt i32 %value.addr.011, 16383
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  %incdec.ptr.lcssa = phi i8* [ %incdec.ptr, %while.body ]
  %shr.lcssa = phi i32 [ %shr, %while.body ]
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %target.addr.0.lcssa = phi i8* [ %target, %entry ], [ %incdec.ptr.lcssa, %while.end.loopexit ]
  %value.addr.0.lcssa = phi i32 [ %value, %entry ], [ %shr.lcssa, %while.end.loopexit ]
  %conv1 = trunc i32 %value.addr.0.lcssa to i8
  store i8 %conv1, i8* %target.addr.0.lcssa, align 1
  %incdec.ptr3 = getelementptr inbounds i8, i8* %target.addr.0.lcssa, i64 1
  ret i8* %incdec.ptr3
}
