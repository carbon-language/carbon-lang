; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK: {
; CHECK: loop0(.LBB
; CHECK-NOT: loop0(##.LBB

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define i32 @q6zip_uncompress(i8* %out_buf, i32* %out_buf_size, i8* %in_buf, i32 %in_buf_size, i8* nocapture %dict, i32 %dict_size) nounwind {
entry:
  %0 = bitcast i8* %in_buf to i32*
  %incdec.ptr = getelementptr inbounds i8, i8* %in_buf, i32 4
  %1 = load i32, i32* %0, align 4, !tbaa !0
  %2 = ptrtoint i8* %incdec.ptr to i32
  %and.i = and i32 %2, 31
  %sub.i = sub i32 %2, %and.i
  %3 = inttoptr i32 %sub.i to i8*
  %add.i = add i32 %in_buf_size, 31
  %sub2.i = add i32 %add.i, %and.i
  %div.i = lshr i32 %sub2.i, 5
  %4 = tail call i32 @llvm.hexagon.A2.combine.ll(i32 32, i32 %div.i) nounwind
  %5 = tail call i64 @llvm.hexagon.A4.combineir(i32 32, i32 %4) nounwind
  tail call void asm sideeffect "l2fetch($0,$1)", "r,r,~{memory}"(i8* %3, i64 %5) nounwind, !srcloc !3
  %6 = ptrtoint i8* %out_buf to i32
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %entry
  %i.02.i = phi i32 [ 0, %entry ], [ %inc.i, %for.body.i ]
  %addr.addr.01.i = phi i32 [ %6, %entry ], [ %add.i14, %for.body.i ]
  tail call void asm sideeffect "dczeroa($0)", "r"(i32 %addr.addr.01.i) nounwind, !srcloc !4
  %add.i14 = add i32 %addr.addr.01.i, 32
  %inc.i = add i32 %i.02.i, 1
  %exitcond.i = icmp eq i32 %inc.i, 128
  br i1 %exitcond.i, label %while.cond.preheader, label %for.body.i

while.cond.preheader:                             ; preds = %for.body.i
  %and = and i32 %1, 3
  switch i32 %and, label %infloop.preheader [
    i32 0, label %exit_inflate.split
    i32 2, label %if.then.preheader
  ]

if.then.preheader:                                ; preds = %while.cond.preheader
  br label %if.then

infloop.preheader:                                ; preds = %while.cond.preheader
  br label %infloop

if.then:                                          ; preds = %if.then.preheader, %if.then
  tail call void @llvm.prefetch(i8* %incdec.ptr, i32 0, i32 3, i32 1)
  br label %if.then

exit_inflate.split:                               ; preds = %while.cond.preheader
  ret i32 0

infloop:                                          ; preds = %infloop.preheader, %infloop
  br label %infloop
}

declare void @llvm.prefetch(i8* nocapture, i32, i32, i32) nounwind

declare i64 @llvm.hexagon.A4.combineir(i32, i32) nounwind readnone

declare i32 @llvm.hexagon.A2.combine.ll(i32, i32) nounwind readnone

!0 = !{!"long", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{i32 18362}
!4 = !{i32 18893}
