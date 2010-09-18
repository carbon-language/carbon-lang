; RUN: llc < %s -march=ptx | FileCheck %s

define void @t1() {
;CHECK: exit;
	ret void
}
