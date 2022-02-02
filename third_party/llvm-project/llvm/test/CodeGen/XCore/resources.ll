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
declare i32 @llvm.xcore.testct.p1i8(i8 addrspace(1)* %r)
declare i32 @llvm.xcore.testwct.p1i8(i8 addrspace(1)* %r)
declare void @llvm.xcore.setd.p1i8(i8 addrspace(1)* %r, i32 %value)
declare void @llvm.xcore.setc.p1i8(i8 addrspace(1)* %r, i32 %value)
declare i32 @llvm.xcore.inshr.p1i8(i8 addrspace(1)* %r, i32 %value)
declare i32 @llvm.xcore.outshr.p1i8(i8 addrspace(1)* %r, i32 %value)
declare void @llvm.xcore.clrpt.p1i8(i8 addrspace(1)* %r)
declare void @llvm.xcore.setpt.p1i8(i8 addrspace(1)* %r, i32 %value)
declare i32 @llvm.xcore.getts.p1i8(i8 addrspace(1)* %r)
declare void @llvm.xcore.syncr.p1i8(i8 addrspace(1)* %r)
declare void @llvm.xcore.settw.p1i8(i8 addrspace(1)* %r, i32 %value)
declare void @llvm.xcore.setv.p1i8(i8 addrspace(1)* %r, i8* %p)
declare void @llvm.xcore.setev.p1i8(i8 addrspace(1)* %r, i8* %p)
declare void @llvm.xcore.edu.p1i8(i8 addrspace(1)* %r)
declare void @llvm.xcore.eeu.p1i8(i8 addrspace(1)* %r)
declare void @llvm.xcore.setclk.p1i8.p1i8(i8 addrspace(1)* %a, i8 addrspace(1)* %b)
declare void @llvm.xcore.setrdy.p1i8.p1i8(i8 addrspace(1)* %a, i8 addrspace(1)* %b)
declare void @llvm.xcore.setpsc.p1i8(i8 addrspace(1)* %r, i32 %value)
declare i32 @llvm.xcore.peek.p1i8(i8 addrspace(1)* %r)
declare i32 @llvm.xcore.endin.p1i8(i8 addrspace(1)* %r)

define i8 addrspace(1)* @getr() {
; CHECK-LABEL: getr:
; CHECK: getr r0, 5
	%result = call i8 addrspace(1)* @llvm.xcore.getr.p1i8(i32 5)
	ret i8 addrspace(1)* %result
}

define void @freer(i8 addrspace(1)* %r) {
; CHECK-LABEL: freer:
; CHECK: freer res[r0]
	call void @llvm.xcore.freer.p1i8(i8 addrspace(1)* %r)
	ret void
}

define i32 @in(i8 addrspace(1)* %r) {
; CHECK-LABEL: in:
; CHECK: in r0, res[r0]
	%result = call i32 @llvm.xcore.in.p1i8(i8 addrspace(1)* %r)
	ret i32 %result
}

define i32 @int(i8 addrspace(1)* %r) {
; CHECK-LABEL: int:
; CHECK: int r0, res[r0]
	%result = call i32 @llvm.xcore.int.p1i8(i8 addrspace(1)* %r)
	ret i32 %result
}

define i32 @inct(i8 addrspace(1)* %r) {
; CHECK-LABEL: inct:
; CHECK: inct r0, res[r0]
	%result = call i32 @llvm.xcore.inct.p1i8(i8 addrspace(1)* %r)
	ret i32 %result
}

define void @out(i8 addrspace(1)* %r, i32 %value) {
; CHECK-LABEL: out:
; CHECK: out res[r0], r1
	call void @llvm.xcore.out.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret void
}

define void @outt(i8 addrspace(1)* %r, i32 %value) {
; CHECK-LABEL: outt:
; CHECK: outt res[r0], r1
	call void @llvm.xcore.outt.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret void
}

define void @outct(i8 addrspace(1)* %r, i32 %value) {
; CHECK-LABEL: outct:
; CHECK: outct res[r0], r1
	call void @llvm.xcore.outct.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret void
}

define void @outcti(i8 addrspace(1)* %r) {
; CHECK-LABEL: outcti:
; CHECK: outct res[r0], 11
	call void @llvm.xcore.outct.p1i8(i8 addrspace(1)* %r, i32 11)
	ret void
}

define void @chkct(i8 addrspace(1)* %r, i32 %value) {
; CHECK-LABEL: chkct:
; CHECK: chkct res[r0], r1
	call void @llvm.xcore.chkct.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret void
}

define void @chkcti(i8 addrspace(1)* %r) {
; CHECK-LABEL: chkcti:
; CHECK: chkct res[r0], 11
	call void @llvm.xcore.chkct.p1i8(i8 addrspace(1)* %r, i32 11)
	ret void
}

define void @setd(i8 addrspace(1)* %r, i32 %value) {
; CHECK-LABEL: setd:
; CHECK: setd res[r0], r1
	call void @llvm.xcore.setd.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret void
}

define void @setc(i8 addrspace(1)* %r, i32 %value) {
; CHECK-LABEL: setc:
; CHECK: setc res[r0], r1
	call void @llvm.xcore.setc.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret void
}

define void @setci(i8 addrspace(1)* %r) {
; CHECK-LABEL: setci:
; CHECK: setc res[r0], 2
	call void @llvm.xcore.setc.p1i8(i8 addrspace(1)* %r, i32 2)
	ret void
}

define i32 @inshr(i32 %value, i8 addrspace(1)* %r) {
; CHECK-LABEL: inshr:
; CHECK: inshr r0, res[r1]
	%result = call i32 @llvm.xcore.inshr.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret i32 %result
}

define i32 @outshr(i32 %value, i8 addrspace(1)* %r) {
; CHECK-LABEL: outshr:
; CHECK: outshr res[r1], r0
	%result = call i32 @llvm.xcore.outshr.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret i32 %result
}

define void @clrpt(i8 addrspace(1)* %r) {
; CHECK-LABEL: clrpt:
; CHECK: clrpt res[r0]
	call void @llvm.xcore.clrpt.p1i8(i8 addrspace(1)* %r)
	ret void
}

define void @setpt(i8 addrspace(1)* %r, i32 %value) {
; CHECK-LABEL: setpt:
; CHECK: setpt res[r0], r1
	call void @llvm.xcore.setpt.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret void
}

define i32 @getts(i8 addrspace(1)* %r) {
; CHECK-LABEL: getts:
; CHECK: getts r0, res[r0]
	%result = call i32 @llvm.xcore.getts.p1i8(i8 addrspace(1)* %r)
	ret i32 %result
}

define void @syncr(i8 addrspace(1)* %r) {
; CHECK-LABEL: syncr:
; CHECK: syncr res[r0]
	call void @llvm.xcore.syncr.p1i8(i8 addrspace(1)* %r)
	ret void
}

define void @settw(i8 addrspace(1)* %r, i32 %value) {
; CHECK-LABEL: settw:
; CHECK: settw res[r0], r1
	call void @llvm.xcore.settw.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret void
}

define void @setv(i8 addrspace(1)* %r, i8* %p) {
; CHECK-LABEL: setv:
; CHECK: mov r11, r1
; CHECK-NEXT: setv res[r0], r11
	call void @llvm.xcore.setv.p1i8(i8 addrspace(1)* %r, i8* %p)
	ret void
}

define void @setev(i8 addrspace(1)* %r, i8* %p) {
; CHECK-LABEL: setev:
; CHECK: mov r11, r1
; CHECK-NEXT: setev res[r0], r11
	call void @llvm.xcore.setev.p1i8(i8 addrspace(1)* %r, i8* %p)
	ret void
}

define void @edu(i8 addrspace(1)* %r) {
; CHECK-LABEL: edu:
; CHECK: edu res[r0]
	call void @llvm.xcore.edu.p1i8(i8 addrspace(1)* %r)
	ret void
}

define void @eeu(i8 addrspace(1)* %r) {
; CHECK-LABEL: eeu:
; CHECK: eeu res[r0]
	call void @llvm.xcore.eeu.p1i8(i8 addrspace(1)* %r)
	ret void
}

define void @setclk(i8 addrspace(1)* %a, i8 addrspace(1)* %b) {
; CHECK: setclk
; CHECK: setclk res[r0], r1
	call void @llvm.xcore.setclk.p1i8.p1i8(i8 addrspace(1)* %a, i8 addrspace(1)* %b)
	ret void
}

define void @setrdy(i8 addrspace(1)* %a, i8 addrspace(1)* %b) {
; CHECK: setrdy
; CHECK: setrdy res[r0], r1
	call void @llvm.xcore.setrdy.p1i8.p1i8(i8 addrspace(1)* %a, i8 addrspace(1)* %b)
	ret void
}

define void @setpsc(i8 addrspace(1)* %r, i32 %value) {
; CHECK: setpsc
; CHECK: setpsc res[r0], r1
	call void @llvm.xcore.setpsc.p1i8(i8 addrspace(1)* %r, i32 %value)
	ret void
}

define i32 @peek(i8 addrspace(1)* %r) {
; CHECK-LABEL: peek:
; CHECK: peek r0, res[r0]
	%result = call i32 @llvm.xcore.peek.p1i8(i8 addrspace(1)* %r)
	ret i32 %result
}

define i32 @endin(i8 addrspace(1)* %r) {
; CHECK-LABEL: endin:
; CHECK: endin r0, res[r0]
	%result = call i32 @llvm.xcore.endin.p1i8(i8 addrspace(1)* %r)
	ret i32 %result
}

define i32 @testct(i8 addrspace(1)* %r) {
; CHECK-LABEL: testct:
; CHECK: testct r0, res[r0]
	%result = call i32 @llvm.xcore.testct.p1i8(i8 addrspace(1)* %r)
	ret i32 %result
}

define i32 @testwct(i8 addrspace(1)* %r) {
; CHECK-LABEL: testwct:
; CHECK: testwct r0, res[r0]
	%result = call i32 @llvm.xcore.testwct.p1i8(i8 addrspace(1)* %r)
	ret i32 %result
}
