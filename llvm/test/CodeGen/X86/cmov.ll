; RUN: llc < %s -march=x86-64 | FileCheck %s

define i32 @test1(i32 %x, i32 %n, i32 %w, i32* %vp) nounwind readnone {
entry:
; CHECK: test1:
; CHECK: btl
; CHECK-NEXT: movl	$12, %eax
; CHECK-NEXT: cmovae	(%rcx), %eax
; CHECK-NEXT: ret

	%0 = lshr i32 %x, %n		; <i32> [#uses=1]
	%1 = and i32 %0, 1		; <i32> [#uses=1]
	%toBool = icmp eq i32 %1, 0		; <i1> [#uses=1]
        %v = load i32* %vp
	%.0 = select i1 %toBool, i32 %v, i32 12		; <i32> [#uses=1]
	ret i32 %.0
}
define i32 @test2(i32 %x, i32 %n, i32 %w, i32* %vp) nounwind readnone {
entry:
; CHECK: test2:
; CHECK: btl
; CHECK-NEXT: movl	$12, %eax
; CHECK-NEXT: cmovb	(%rcx), %eax
; CHECK-NEXT: ret

	%0 = lshr i32 %x, %n		; <i32> [#uses=1]
	%1 = and i32 %0, 1		; <i32> [#uses=1]
	%toBool = icmp eq i32 %1, 0		; <i1> [#uses=1]
        %v = load i32* %vp
	%.0 = select i1 %toBool, i32 12, i32 %v		; <i32> [#uses=1]
	ret i32 %.0
}


declare void @bar(i64) nounwind

define void @test3(i64 %a, i64 %b, i1 %p) nounwind {
; CHECK: test3:
; CHECK:      cmovne  %edi, %esi
; CHECK-NEXT: movl    %esi, %edi

  %c = trunc i64 %a to i32
  %d = trunc i64 %b to i32
  %e = select i1 %p, i32 %c, i32 %d
  %f = zext i32 %e to i64
  call void @bar(i64 %f)
  ret void
}
