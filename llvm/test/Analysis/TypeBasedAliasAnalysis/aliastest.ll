; RUN: opt < %s -tbaa -basicaa -gvn -S | FileCheck %s

; Test that basic alias queries work.

; CHECK: @test0_yes
; CHECK: add i8 %x, %x
define i8 @test0_yes(i8* %a, i8* %b) nounwind {
  %x = load i8* %a, !tbaa !1
  store i8 0, i8* %b, !tbaa !2
  %y = load i8* %a, !tbaa !1
  %z = add i8 %x, %y
  ret i8 %z
}

; CHECK: @test0_no
; CHECK: add i8 %x, %y
define i8 @test0_no(i8* %a, i8* %b) nounwind {
  %x = load i8* %a, !tbaa !3
  store i8 0, i8* %b, !tbaa !4
  %y = load i8* %a, !tbaa !3
  %z = add i8 %x, %y
  ret i8 %z
}

; Test that basic invariant-memory queries work.

; CHECK: @test1_yes
; CHECK: add i8 %x, %x
define i8 @test1_yes(i8* %a, i8* %b) nounwind {
  %x = load i8* %a, !tbaa !5
  store i8 0, i8* %b
  %y = load i8* %a, !tbaa !5
  %z = add i8 %x, %y
  ret i8 %z
}

; CHECK: @test1_no
; CHECK: add i8 %x, %y
define i8 @test1_no(i8* %a, i8* %b) nounwind {
  %x = load i8* %a, !tbaa !6
  store i8 0, i8* %b
  %y = load i8* %a, !tbaa !6
  %z = add i8 %x, %y
  ret i8 %z
}

; Root note.
!0 = metadata !{ }
; Some type.
!1 = metadata !{metadata !7, metadata !7, i64 0}
; Some other non-aliasing type.
!2 = metadata !{metadata !8, metadata !8, i64 0}

; Some type.
!3 = metadata !{metadata !9, metadata !9, i64 0}
; Some type in a different type system.
!4 = metadata !{metadata !10, metadata !10, i64 0}

; Invariant memory.
!5 = metadata !{metadata !11, metadata !11, i64 0, i1 1}
; Not invariant memory.
!6 = metadata !{metadata !11, metadata !11, i64 0, i1 0}
!7 = metadata !{ metadata !"foo", metadata !0 }
!8 = metadata !{ metadata !"bar", metadata !0 }
!9 = metadata !{ metadata !"foo", metadata !0 }
!10 = metadata !{ metadata !"bar", metadata !"different" }
!11 = metadata !{ metadata !"qux", metadata !0}
