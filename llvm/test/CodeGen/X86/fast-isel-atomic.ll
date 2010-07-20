; RUN: llc < %s -O0 -march=x86-64
; rdar://8204072
; PR7652

@sc = external global i8
@uc = external global i8

declare i8 @llvm.atomic.load.and.i8.p0i8(i8* nocapture, i8) nounwind

define void @test_fetch_and_op() nounwind {
entry:
  %tmp40 = call i8 @llvm.atomic.load.and.i8.p0i8(i8* @sc, i8 11) ; <i8> [#uses=1]
  store i8 %tmp40, i8* @sc
  %tmp41 = call i8 @llvm.atomic.load.and.i8.p0i8(i8* @uc, i8 11) ; <i8> [#uses=1]
  store i8 %tmp41, i8* @uc
  ret void
}
