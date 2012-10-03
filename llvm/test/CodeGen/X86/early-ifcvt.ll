; RUN: llc < %s -x86-early-ifcvt -stress-early-ifcvt | FileCheck %s
target triple = "x86_64-apple-macosx10.8.0"

; CHECK: mm2
define i32 @mm2(i32* nocapture %p, i32 %n) nounwind uwtable readonly ssp {
entry:
  br label %do.body

; CHECK: do.body
; Loop body has no branches before the backedge.
; CHECK-NOT: LBB
do.body:
  %max.0 = phi i32 [ 0, %entry ], [ %max.1, %do.cond ]
  %min.0 = phi i32 [ 0, %entry ], [ %min.1, %do.cond ]
  %n.addr.0 = phi i32 [ %n, %entry ], [ %dec, %do.cond ]
  %p.addr.0 = phi i32* [ %p, %entry ], [ %incdec.ptr, %do.cond ]
  %incdec.ptr = getelementptr inbounds i32* %p.addr.0, i64 1
  %0 = load i32* %p.addr.0, align 4
  %cmp = icmp sgt i32 %0, %max.0
  br i1 %cmp, label %do.cond, label %if.else

if.else:
  %cmp1 = icmp slt i32 %0, %min.0
  %.min.0 = select i1 %cmp1, i32 %0, i32 %min.0
  br label %do.cond

do.cond:
  %max.1 = phi i32 [ %0, %do.body ], [ %max.0, %if.else ]
  %min.1 = phi i32 [ %min.0, %do.body ], [ %.min.0, %if.else ]
; CHECK: decl %esi
; CHECK: jne LBB
  %dec = add i32 %n.addr.0, -1
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %do.end, label %do.body

do.end:
  %sub = sub nsw i32 %max.1, %min.1
  ret i32 %sub
}

; CHECK: multipreds
; Deal with alternative tail predecessors
; CHECK-NOT: LBB
; CHECK: cmov
; CHECK-NOT: LBB
; CHECK: cmov
; CHECK-NOT: LBB
; CHECK: fprintf

define void @multipreds(i32 %sw) nounwind uwtable ssp {
entry:
  switch i32 %sw, label %if.then29 [
    i32 0, label %if.then37
    i32 127, label %if.end41
  ]

if.then29:
  br label %if.end41

if.then37:
  br label %if.end41

if.end41:
  %exit_status.0 = phi i32 [ 2, %if.then29 ], [ 0, %if.then37 ], [ 66, %entry ]
  call void (...)* @fprintf(i32 %exit_status.0) nounwind
  unreachable
}

declare void @fprintf(...) nounwind
