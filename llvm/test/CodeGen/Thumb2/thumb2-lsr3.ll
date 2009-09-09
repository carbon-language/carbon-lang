; RUN: llc < %s -march=thumb -mattr=+thumb2

define i1 @test1(i64 %poscnt, i32 %work) {
entry:
; CHECK: rrx r0, r0
; CHECK: lsrs.w r1, r1, #1
	%0 = lshr i64 %poscnt, 1
	%1 = icmp eq i64 %0, 0
	ret i1 %1
}

define i1 @test2(i64 %poscnt, i32 %work) {
entry:
; CHECK: rrx r0, r0
; CHECK: asrs.w r1, r1, #1
	%0 = ashr i64 %poscnt, 1
	%1 = icmp eq i64 %0, 0
	ret i1 %1
}
