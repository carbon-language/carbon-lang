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

declare void @llvm.ptx.bar.sync(i32 %i)
