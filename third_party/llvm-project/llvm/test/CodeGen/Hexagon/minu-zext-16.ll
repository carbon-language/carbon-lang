; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: minu

define zeroext i16 @f(i16* noalias nocapture %src) nounwind readonly {
entry:
  %arrayidx = getelementptr inbounds i16, i16* %src, i32 1
  %0 = load i16, i16* %arrayidx, align 1
  %cmp = icmp ult i16 %0, 32767
  %. = select i1 %cmp, i16 %0, i16 32767
  ret i16 %.
}
