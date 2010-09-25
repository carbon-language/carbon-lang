; RUN: llc < %s -march=ptx | FileCheck %s

define ptx_kernel void @t1() {
;CHECK: exit;
	ret void
}
