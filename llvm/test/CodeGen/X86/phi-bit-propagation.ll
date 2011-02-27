; RUN: llc < %s -march=x86-64 | FileCheck %s

%"class.std::bitset" = type { [8 x i8] }

define zeroext i1 @_Z3fooPjmS_mRSt6bitsetILm32EE(i32* nocapture %a, i64 %asize, i32* nocapture %b, i64 %bsize, %"class.std::bitset"* %bits) nounwind readonly ssp noredzone {
entry:
  %tmp.i.i.i.i = bitcast %"class.std::bitset"* %bits to i64*
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %conv = zext i32 %0 to i64
  %cmp = icmp eq i64 %conv, %bsize
  br i1 %cmp, label %return, label %for.body

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32* %b, i64 %conv
  %tmp5 = load i32* %arrayidx, align 4
  %conv6 = zext i32 %tmp5 to i64
  %rem.i.i.i.i = and i64 %conv6, 63
  %tmp3.i = load i64* %tmp.i.i.i.i, align 8
  %shl.i.i = shl i64 1, %rem.i.i.i.i
  %and.i = and i64 %shl.i.i, %tmp3.i
  %cmp.i = icmp eq i64 %and.i, 0
  br i1 %cmp.i, label %for.inc, label %return

for.inc:                                          ; preds = %for.body
  %inc = add i32 %0, 1
  br label %for.cond

return:                                           ; preds = %for.body, %for.cond
; CHECK-NOT: and
  %retval.0 = phi i1 [ true, %for.body ], [ false, %for.cond ]
  ret i1 %retval.0
}
