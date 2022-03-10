; RUN: llc -march=xcore < %s | FileCheck %s

declare i32 @llvm.xcore.int.p1i8(i8 addrspace(1)* %r)
declare i32 @llvm.xcore.inct.p1i8(i8 addrspace(1)* %r)
declare i32 @llvm.xcore.testct.p1i8(i8 addrspace(1)* %r)
declare i32 @llvm.xcore.testwct.p1i8(i8 addrspace(1)* %r)
declare i32 @llvm.xcore.getts.p1i8(i8 addrspace(1)* %r)
declare void @llvm.xcore.outt.p1i8(i8 addrspace(1)* %r, i32 %value)
declare void @llvm.xcore.outct.p1i8(i8 addrspace(1)* %r, i32 %value)
declare void @llvm.xcore.chkct.p1i8(i8 addrspace(1)* %r, i32 %value)
declare void @llvm.xcore.setpt.p1i8(i8 addrspace(1)* %r, i32 %value)

define i32 @int(i8 addrspace(1)* %r) nounwind {
; CHECK-LABEL: int:
; CHECK: int r0, res[r0]
; CHECK-NEXT: retsp 0
	%result = call i32 @llvm.xcore.int.p1i8(i8 addrspace(1)* %r)
	%trunc = and i32 %result, 255
	ret i32 %trunc
}

define i32 @inct(i8 addrspace(1)* %r) nounwind {
; CHECK-LABEL: inct:
; CHECK: inct r0, res[r0]
; CHECK-NEXT: retsp 0
	%result = call i32 @llvm.xcore.inct.p1i8(i8 addrspace(1)* %r)
	%trunc = and i32 %result, 255
	ret i32 %trunc
}

define i32 @testct(i8 addrspace(1)* %r) nounwind {
; CHECK-LABEL: testct:
; CHECK: testct r0, res[r0]
; CHECK-NEXT: retsp 0
	%result = call i32 @llvm.xcore.testct.p1i8(i8 addrspace(1)* %r)
	%trunc = and i32 %result, 1
	ret i32 %trunc
}

define i32 @testwct(i8 addrspace(1)* %r) nounwind {
; CHECK-LABEL: testwct:
; CHECK: testwct r0, res[r0]
; CHECK-NEXT: retsp 0
	%result = call i32 @llvm.xcore.testwct.p1i8(i8 addrspace(1)* %r)
	%trunc = and i32 %result, 7
	ret i32 %trunc
}

define i32 @getts(i8 addrspace(1)* %r) nounwind {
; CHECK-LABEL: getts:
; CHECK: getts r0, res[r0]
; CHECK-NEXT: retsp 0
	%result = call i32 @llvm.xcore.getts.p1i8(i8 addrspace(1)* %r)
	%trunc = and i32 %result, 65535
	ret i32 %result
}

define void @outt(i8 addrspace(1)* %r, i32 %value) nounwind {
; CHECK-LABEL: outt:
; CHECK-NOT: zext
; CHECK: outt res[r0], r1
; CHECK-NEXT: retsp 0
	%trunc = and i32 %value, 255
	call void @llvm.xcore.outt.p1i8(i8 addrspace(1)* %r, i32 %trunc)
	ret void
}

define void @outct(i8 addrspace(1)* %r, i32 %value) nounwind {
; CHECK-LABEL: outct:
; CHECK-NOT: zext
; CHECK: outct res[r0], r1
	%trunc = and i32 %value, 255
	call void @llvm.xcore.outct.p1i8(i8 addrspace(1)* %r, i32 %trunc)
	ret void
}

define void @chkct(i8 addrspace(1)* %r, i32 %value) nounwind {
; CHECK-LABEL: chkct:
; CHECK-NOT: zext
; CHECK: chkct res[r0], r1
	%trunc = and i32 %value, 255
	call void @llvm.xcore.chkct.p1i8(i8 addrspace(1)* %r, i32 %trunc)
	ret void
}

define void @setpt(i8 addrspace(1)* %r, i32 %value) nounwind {
; CHECK-LABEL: setpt:
; CHECK-NOT: zext
; CHECK: setpt res[r0], r1
	%trunc = and i32 %value, 65535
	call void @llvm.xcore.setpt.p1i8(i8 addrspace(1)* %r, i32 %trunc)
	ret void
}
