; RUN: llc -march=x86-64 < %s | FileCheck %s

; LSR should compute the correct starting values for this loop. Note that
; it's not necessarily LSR's job to compute loop exit expressions; that's
; indvars' job.
; CHECK: movl  $12
; CHECK: movl  $42

define i32 @real_symmetric_eigen(i32 %n) nounwind {
while.body127:                                    ; preds = %while.cond122
  br label %while.cond141

while.cond141:                                    ; preds = %while.cond141, %while.body127
  %0 = phi i32 [ 7, %while.body127 ], [ %indvar.next67, %while.cond141 ] ; <i32> [#uses=3]
  %indvar.next67 = add i32 %0, 1                  ; <i32> [#uses=1]
  %t = icmp slt i32 %indvar.next67, %n
  br i1 %t, label %if.then171, label %while.cond141

if.then171:                                       ; preds = %while.cond141
  %mul150 = mul i32 %0, %0                 ; <i32> [#uses=1]
  %add174 = add i32 %mul150, %0                 ; <i32> [#uses=1]
  ret i32 %add174
}
