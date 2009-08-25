; RUN: llvm-as < %s | llc -march=x86 | FileCheck %s

define i32 @test1(i32 %x, i32 %n, i32 %w, i32 %v) nounwind readnone {
entry:
; CHECK: test1:
; CHECK: btl
; CHECK-NEXT: movl	$12, %eax
; CHECK-NEXT: cmovae	16(%esp), %eax
; CHECK-NEXT: ret

	%0 = lshr i32 %x, %n		; <i32> [#uses=1]
	%1 = and i32 %0, 1		; <i32> [#uses=1]
	%toBool = icmp eq i32 %1, 0		; <i1> [#uses=1]
	%.0 = select i1 %toBool, i32 %v, i32 12		; <i32> [#uses=1]
	ret i32 %.0
}
define i32 @test2(i32 %x, i32 %n, i32 %w, i32 %v) nounwind readnone {
entry:
; CHECK: test2:
; CHECK: btl
; CHECK-NEXT: movl	$12, %eax
; CHECK-NEXT: cmovb	16(%esp), %eax
; CHECK-NEXT: ret

	%0 = lshr i32 %x, %n		; <i32> [#uses=1]
	%1 = and i32 %0, 1		; <i32> [#uses=1]
	%toBool = icmp eq i32 %1, 0		; <i1> [#uses=1]
	%.0 = select i1 %toBool, i32 12, i32 %v		; <i32> [#uses=1]
	ret i32 %.0
}
