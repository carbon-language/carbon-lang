; RUN: llc < %s -march=ptx32 | FileCheck %s

define ptx_device void @test_bra_direct() {
; CHECK: bra $L__BB0_1;
entry:
	br label %loop
loop:
	br label %loop
}

define ptx_device i32 @test_bra_cond_direct(i32 %x, i32 %y) {
entry:
; CHECK: setp.le.u32 p0, r1, r2
	%p = icmp ugt i32 %x, %y
; CHECK-NEXT: @p0 bra
; CHECK-NOT: bra
	br i1 %p, label %clause.if, label %clause.else
clause.if:
; CHECK: mov.u32 r0, r1
	ret i32 %x
clause.else:
; CHECK: mov.u32 r0, r2
	ret i32 %y
}
