; RUN: llc -march=xcore < %s | FileCheck %s

declare i32 @llvm.xcore.int.p1i8(i8 addrspace(1)* %r)
declare i32 @llvm.xcore.inct.p1i8(i8 addrspace(1)* %r)
declare i32 @llvm.xcore.testct.p1i8(i8 addrspace(1)* %r)
declare i32 @llvm.xcore.testwct.p1i8(i8 addrspace(1)* %r)
declare i32 @llvm.xcore.getts.p1i8(i8 addrspace(1)* %r)

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
