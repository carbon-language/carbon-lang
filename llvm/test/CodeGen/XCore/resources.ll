; RUN: llc -march=xcore < %s | FileCheck %s

declare i8 addrspace(1)* @llvm.xcore.getr.p1i8(i32 %type)
declare void @llvm.xcore.freer.p1i8(i8 addrspace(1)* %r)
declare i32 @llvm.xcore.in.p1i8(i8 addrspace(1)* %r)
declare i32 @llvm.xcore.int.p1i8(i8 addrspace(1)* %r)
declare i32 @llvm.xcore.inct.p1i8(i8 addrspace(1)* %r)
declare void @llvm.xcore.out.p1i8(i8 addrspace(1)* %r, i32 %value)
declare void @llvm.xcore.outt.p1i8(i8 addrspace(1)* %r, i32 %value)
declare void @llvm.xcore.outct.p1i8(i8 addrspace(1)* %r, i32 %value)
declare void @llvm.xcore.chkct.p1i8(i8 addrspace(1)* %r, i32 %value)
declare void @llvm.xcore.setd.p1i8(i8 addrspace(1)* %r, i32 %value)
declare void @llvm.xcore.setc.p1i8(i8 addrspace(1)* %r, i32 %value)

define i8 addrspace(1)* @getr() {
; CHECK: getr:
; CHECK: getr r0, 5
	%result = call i8 addrspace(1)* @llvm.xcore.getr.p1i8(i32 5)
	ret i8 addrspace(1)* %result
}

define void @freer(i8 addrspace(1)* %r) {
; CHECK: freer:
; CHECK: freer res[r0]
	call void @llvm.xcore.freer.p1i8(i8 addrspace(1)* %r)
	ret void
}

define i32 @in(i8 addrspace(1)* %r) {
; CHECK: in:
; CHECK: in r0, res[r0]
	%result = call i32 @llvm.xcore.in.p1i8(i8 addrspace(1)* %r)
	ret i32 %result
}

define i32 @int(i8 addrspace(1)* %r) {
; CHECK: int:
; CHECK: int r0, res[r0]
	%result = call i32 @llvm.xcore.int.p1i8(i8 addrspace(1)* %r)
	ret i32 %result
}

define i32 @inct(i8 addrspace(1)* %r) {
; CHECK: inct:
; CHECK: inct r0, res[r0]
	%result = call i32 @llvm.xcore.inct.p1i8(i8 addrspace(1)* %r)
	ret i32 %result
}

define void @out(i8 addrspace(1)* %r, i32 %value) {
; CHECK: out:
; CHECK: out res[r0], r1
	call void @llvm.xcore.out.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret void
}

define void @outt(i8 addrspace(1)* %r, i32 %value) {
; CHECK: outt:
; CHECK: outt res[r0], r1
	call void @llvm.xcore.outt.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret void
}

define void @outct(i8 addrspace(1)* %r, i32 %value) {
; CHECK: outct:
; CHECK: outct res[r0], r1
	call void @llvm.xcore.outct.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret void
}

define void @outcti(i8 addrspace(1)* %r) {
; CHECK: outcti:
; CHECK: outct res[r0], 11
	call void @llvm.xcore.outct.p1i8(i8 addrspace(1)* %r, i32 11)
	ret void
}

define void @chkct(i8 addrspace(1)* %r, i32 %value) {
; CHECK: chkct:
; CHECK: chkct res[r0], r1
	call void @llvm.xcore.chkct.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret void
}

define void @chkcti(i8 addrspace(1)* %r) {
; CHECK: chkcti:
; CHECK: chkct res[r0], 11
	call void @llvm.xcore.chkct.p1i8(i8 addrspace(1)* %r, i32 11)
	ret void
}

define void @setd(i8 addrspace(1)* %r, i32 %value) {
; CHECK: setd:
; CHECK: setd res[r0], r1
	call void @llvm.xcore.setd.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret void
}

define void @setc(i8 addrspace(1)* %r, i32 %value) {
; CHECK: setc:
; CHECK: setc res[r0], r1
	call void @llvm.xcore.setc.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret void
}

define void @setci(i8 addrspace(1)* %r) {
; CHECK: setci:
; CHECK: setc res[r0], 2
	call void @llvm.xcore.setc.p1i8(i8 addrspace(1)* %r, i32 2)
	ret void
}
