; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; CHECK: word
; CHECK: combine(#0

define void @word(i32* nocapture %a) nounwind {
entry:
  %0 = load i32* %a, align 4, !tbaa !0
  %1 = zext i32 %0 to i64
  tail call void @bar(i64 %1) nounwind
  ret void
}

declare void @bar(i64)

; CHECK: halfword
; CHECK: combine(#0

define void @halfword(i16* nocapture %a) nounwind {
entry:
  %0 = load i16* %a, align 2, !tbaa !3
  %1 = zext i16 %0 to i64
  %add.ptr = getelementptr inbounds i16* %a, i32 1
  %2 = load i16* %add.ptr, align 2, !tbaa !3
  %3 = zext i16 %2 to i64
  %4 = shl nuw nsw i64 %3, 16
  %ins = or i64 %4, %1
  tail call void @bar(i64 %ins) nounwind
  ret void
}

; CHECK: byte
; CHECK: combine(#0

define void @byte(i8* nocapture %a) nounwind {
entry:
  %0 = load i8* %a, align 1, !tbaa !1
  %1 = zext i8 %0 to i64
  %add.ptr = getelementptr inbounds i8* %a, i32 1
  %2 = load i8* %add.ptr, align 1, !tbaa !1
  %3 = zext i8 %2 to i64
  %4 = shl nuw nsw i64 %3, 8
  %ins = or i64 %4, %1
  tail call void @bar(i64 %ins) nounwind
  ret void
}

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
!3 = metadata !{metadata !"short", metadata !1}
