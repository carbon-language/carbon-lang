; RUN: llc < %s -march=ptx | FileCheck %s

define ptx_device i16 @tid_x() {
; CHECK: mov.u16 rh0, tid.x;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.tid.x()
	ret i16 %x
}

define ptx_device i16 @tid_y() {
; CHECK: mov.u16 rh0, tid.y;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.tid.y()
	ret i16 %x
}

define ptx_device i16 @tid_z() {
; CHECK: mov.u16 rh0, tid.z;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.tid.z()
	ret i16 %x
}

define ptx_device i16 @tid_w() {
; CHECK: mov.u16 rh0, tid.w;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.tid.w()
	ret i16 %x
}

define ptx_device i16 @ntid_x() {
; CHECK: mov.u16 rh0, ntid.x;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.ntid.x()
	ret i16 %x
}

define ptx_device i16 @ntid_y() {
; CHECK: mov.u16 rh0, ntid.y;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.ntid.y()
	ret i16 %x
}

define ptx_device i16 @ntid_z() {
; CHECK: mov.u16 rh0, ntid.z;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.ntid.z()
	ret i16 %x
}

define ptx_device i16 @ntid_w() {
; CHECK: mov.u16 rh0, ntid.w;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.ntid.w()
	ret i16 %x
}

define ptx_device i16 @ctaid_x() {
; CHECK: mov.u16 rh0, ctaid.x;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.ctaid.x()
	ret i16 %x
}

define ptx_device i16 @ctaid_y() {
; CHECK: mov.u16 rh0, ctaid.y;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.ctaid.y()
	ret i16 %x
}

define ptx_device i16 @ctaid_z() {
; CHECK: mov.u16 rh0, ctaid.z;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.ctaid.z()
	ret i16 %x
}

define ptx_device i16 @ctaid_w() {
; CHECK: mov.u16 rh0, ctaid.w;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.ctaid.w()
	ret i16 %x
}

define ptx_device i16 @nctaid_x() {
; CHECK: mov.u16 rh0, nctaid.x;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.nctaid.x()
	ret i16 %x
}

define ptx_device i16 @nctaid_y() {
; CHECK: mov.u16 rh0, nctaid.y;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.nctaid.y()
	ret i16 %x
}

define ptx_device i16 @nctaid_z() {
; CHECK: mov.u16 rh0, nctaid.z;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.nctaid.z()
	ret i16 %x
}

define ptx_device i16 @nctaid_w() {
; CHECK: mov.u16 rh0, nctaid.w;
; CHECK-NEXT: ret;
	%x = call i16 @llvm.ptx.read.nctaid.w()
	ret i16 %x
}

define ptx_device void @bar_sync() {
; CHECK: bar.sync 0
; CHECK-NEXT: ret;
	call void @llvm.ptx.bar.sync(i32 0)
	ret void
}

declare i16 @llvm.ptx.read.tid.x()
declare i16 @llvm.ptx.read.tid.y()
declare i16 @llvm.ptx.read.tid.z()
declare i16 @llvm.ptx.read.tid.w()
declare i16 @llvm.ptx.read.ntid.x()
declare i16 @llvm.ptx.read.ntid.y()
declare i16 @llvm.ptx.read.ntid.z()
declare i16 @llvm.ptx.read.ntid.w()
declare i16 @llvm.ptx.read.ctaid.x()
declare i16 @llvm.ptx.read.ctaid.y()
declare i16 @llvm.ptx.read.ctaid.z()
declare i16 @llvm.ptx.read.ctaid.w()
declare i16 @llvm.ptx.read.nctaid.x()
declare i16 @llvm.ptx.read.nctaid.y()
declare i16 @llvm.ptx.read.nctaid.z()
declare i16 @llvm.ptx.read.nctaid.w()

declare void @llvm.ptx.bar.sync(i32 %i)
