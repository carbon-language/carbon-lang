; RUN: llc < %s -march=ptx32 | FileCheck %s

define ptx_device void @t1() {
; CHECK: ret;
; CHECK-NOT: exit;
	ret void
}
