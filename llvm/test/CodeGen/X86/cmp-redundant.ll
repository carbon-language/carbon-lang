; RUN: llc < %s -mtriple=x86_64-apple-darwin10 | FileCheck %s

define i32 @cmp(i32* %aa, i32* %bb) nounwind readnone ssp {
entry:
  %a = load i32* %aa
  %b = load i32* %bb
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %return, label %if.end
; CHECK: cmp:
; CHECK: cmpl
; CHECK: jg

if.end:                                           ; preds = %entry
; CHECK-NOT: cmpl
; CHECK: cmov
  %cmp4 = icmp slt i32 %a, %b
  %. = select i1 %cmp4, i32 2, i32 111
  br label %return

return:                                           ; preds = %if.end, %entry
  %retval.0 = phi i32 [ 1, %entry ], [ %., %if.end ]
  ret i32 %retval.0
}

define i32 @cmp2(i32 %a, i32 %b) nounwind readnone ssp {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %return, label %if.end
; CHECK: cmp2:
; CHECK: cmpl
; CHECK: jg

if.end:                                           ; preds = %entry
; CHECK-NOT: cmpl
; CHECK: cmov
  %cmp4 = icmp slt i32 %a, %b
  %. = select i1 %cmp4, i32 2, i32 111
  br label %return

return:                                           ; preds = %if.end, %entry
  %retval.0 = phi i32 [ 1, %entry ], [ %., %if.end ]
  ret i32 %retval.0
}
