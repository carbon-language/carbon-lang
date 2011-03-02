; RUN: llc < %s -march=ptx -mattr=ptx14 | grep ".version 1.4"
; RUN: llc < %s -march=ptx -mattr=ptx20 | grep ".version 2.0"
; RUN: llc < %s -march=ptx -mattr=ptx21 | grep ".version 2.1"
; RUN: llc < %s -march=ptx -mattr=sm20 | grep ".target sm_20"
; RUN: llc < %s -march=ptx -mattr=sm13 | grep ".target sm_13"

define ptx_device void @t1() {
	ret void
}
