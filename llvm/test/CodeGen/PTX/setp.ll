; RUN: llc < %s -march=ptx | FileCheck %s

define ptx_device i32 @test_setp_eq_u32_rr(i32 %x, i32 %y) {
; CHECK: setp.eq.u32 p0, r1, r2;
; CHECK-NEXT: cvt.u32.pred r0, p0;
; CHECK-NEXT: ret;
	%p = icmp eq i32 %x, %y
	%z = zext i1 %p to i32
	ret i32 %z
}

define ptx_device i32 @test_setp_ne_u32_rr(i32 %x, i32 %y) {
; CHECK: setp.ne.u32 p0, r1, r2;
; CHECK-NEXT: cvt.u32.pred r0, p0;
; CHECK-NEXT: ret;
	%p = icmp ne i32 %x, %y
	%z = zext i1 %p to i32
	ret i32 %z
}

define ptx_device i32 @test_setp_lt_u32_rr(i32 %x, i32 %y) {
; CHECK: setp.lt.u32 p0, r1, r2;
; CHECK-NEXT: cvt.u32.pred r0, p0;
; CHECK-NEXT: ret;
	%p = icmp ult i32 %x, %y
	%z = zext i1 %p to i32
	ret i32 %z
}

define ptx_device i32 @test_setp_le_u32_rr(i32 %x, i32 %y) {
; CHECK: setp.le.u32 p0, r1, r2;
; CHECK-NEXT: cvt.u32.pred r0, p0;
; CHECK-NEXT: ret;
	%p = icmp ule i32 %x, %y
	%z = zext i1 %p to i32
	ret i32 %z
}

define ptx_device i32 @test_setp_gt_u32_rr(i32 %x, i32 %y) {
; CHECK: setp.gt.u32 p0, r1, r2;
; CHECK-NEXT: cvt.u32.pred r0, p0;
; CHECK-NEXT: ret;
	%p = icmp ugt i32 %x, %y
	%z = zext i1 %p to i32
	ret i32 %z
}

define ptx_device i32 @test_setp_ge_u32_rr(i32 %x, i32 %y) {
; CHECK: setp.ge.u32 p0, r1, r2;
; CHECK-NEXT: cvt.u32.pred r0, p0;
; CHECK-NEXT: ret;
	%p = icmp uge i32 %x, %y
	%z = zext i1 %p to i32
	ret i32 %z
}

define ptx_device i32 @test_setp_eq_u32_ri(i32 %x) {
; CHECK: setp.eq.u32 p0, r1, 1;
; CHECK-NEXT: cvt.u32.pred r0, p0;
; CHECK-NEXT: ret;
	%p = icmp eq i32 %x, 1
	%z = zext i1 %p to i32
	ret i32 %z
}

define ptx_device i32 @test_setp_ne_u32_ri(i32 %x) {
; CHECK: setp.ne.u32 p0, r1, 1;
; CHECK-NEXT: cvt.u32.pred r0, p0;
; CHECK-NEXT: ret;
	%p = icmp ne i32 %x, 1
	%z = zext i1 %p to i32
	ret i32 %z
}

define ptx_device i32 @test_setp_lt_u32_ri(i32 %x) {
; CHECK: setp.eq.u32 p0, r1, 0;
; CHECK-NEXT: cvt.u32.pred r0, p0;
; CHECK-NEXT: ret;
	%p = icmp ult i32 %x, 1
	%z = zext i1 %p to i32
	ret i32 %z
}

define ptx_device i32 @test_setp_le_u32_ri(i32 %x) {
; CHECK: setp.lt.u32 p0, r1, 2;
; CHECK-NEXT: cvt.u32.pred r0, p0;
; CHECK-NEXT: ret;
	%p = icmp ule i32 %x, 1
	%z = zext i1 %p to i32
	ret i32 %z
}

define ptx_device i32 @test_setp_gt_u32_ri(i32 %x) {
; CHECK: setp.gt.u32 p0, r1, 1;
; CHECK-NEXT: cvt.u32.pred r0, p0;
; CHECK-NEXT: ret;
	%p = icmp ugt i32 %x, 1
	%z = zext i1 %p to i32
	ret i32 %z
}

define ptx_device i32 @test_setp_ge_u32_ri(i32 %x) {
; CHECK: setp.ne.u32 p0, r1, 0;
; CHECK-NEXT: cvt.u32.pred r0, p0;
; CHECK-NEXT: ret;
	%p = icmp uge i32 %x, 1
	%z = zext i1 %p to i32
	ret i32 %z
}
