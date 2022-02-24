; RUN: llc -mtriple=arm64-unknown-unknown -print-lsr-output < %s 2>&1 | FileCheck %s

declare void @foo(i64)

; Verify that redundant adds aren't inserted by LSR.
; CHECK-LABEL: @bar(
define void @bar(double* %A) {
entry:
  br label %while.cond

while.cond:
; CHECK-LABEL: while.cond:
; CHECK: add i64 %lsr.iv, 1
; CHECK-NOT: add i64 %lsr.iv, 1
; CHECK-LABEL: land.rhs:
  %indvars.iv28 = phi i64 [ %indvars.iv.next29, %land.rhs ], [ 50, %entry ]
  %cmp = icmp sgt i64 %indvars.iv28, 0
  br i1 %cmp, label %land.rhs, label %while.end

land.rhs:
  %indvars.iv.next29 = add nsw i64 %indvars.iv28, -1
  %arrayidx = getelementptr inbounds double, double* %A, i64 %indvars.iv.next29
  %Aload = load double, double* %arrayidx, align 8
  %cmp1 = fcmp oeq double %Aload, 0.000000e+00
  br i1 %cmp1, label %while.cond, label %if.end

while.end:
  %indvars.iv28.lcssa = phi i64 [ %indvars.iv28, %while.cond ]
  tail call void @foo(i64 %indvars.iv28.lcssa)
  br label %if.end

if.end:
  ret void
}
