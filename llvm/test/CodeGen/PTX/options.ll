; RUN: llc < %s -march=ptx -ptx-version=2.0 | grep ".version 2.0"
; RUN: llc < %s -march=ptx -ptx-target=sm_20 | grep ".target sm_20"

define ptx_device void @t1() {
	ret void
}
