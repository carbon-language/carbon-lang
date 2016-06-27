; RUN: opt < %s -loop-vectorize -S | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; CHECK-LABEL: @img2buf
; CHECK: store <4 x i32>
; Function Attrs: nounwind
define void @img2buf(i64 %val, i8* nocapture %buf, i32 %N) local_unnamed_addr #0 {
entry:
  br label %l2

l2:
  br label %for.body57.us

for.body57.us: 
  %indvars.iv24 = phi i64 [ %val, %l2 ], [ %indvars.iv.next25, %for.body57.us ]
  %0 = trunc i64 %indvars.iv24 to i32
  %add77.us = add i32 5, %0
  %mul78.us = shl nsw i32 %add77.us, 2
  %idx.ext79.us = sext i32 %mul78.us to i64
  %add.ptr80.us = getelementptr inbounds i8, i8* %buf, i64 %idx.ext79.us
  %ui32.0.add.ptr80.sroa_cast.us = bitcast i8* %add.ptr80.us to i32*
  store i32 0, i32* %ui32.0.add.ptr80.sroa_cast.us, align 1
  %indvars.iv.next25 = add nsw i64 %indvars.iv24, 1
  %lftr.wideiv26 = trunc i64 %indvars.iv.next25 to i32
  %exitcond27 = icmp eq i32 %lftr.wideiv26, %N
  br i1 %exitcond27, label %l3, label %for.body57.us

l3: 
  ret void
}

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="ppc64" "target-features"="+altivec,-bpermd,-crypto,-direct-move,-extdiv,-power8-vector,-qpx,-vsx" "unsafe-fp-math"="false" "use-soft-float"="false" }

