; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: minu

define zeroext i8 @f(i8* noalias nocapture %src) nounwind readonly {
entry:
  %arrayidx = getelementptr inbounds i8, i8* %src, i32 1
  %0 = load i8, i8* %arrayidx, align 1
  %cmp = icmp ult i8 %0, 127
  %. = select i1 %cmp, i8 %0, i8 127
  ret i8 %.
}
