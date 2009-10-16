; RUN: llc < %s -march=x86 -mattr=+sse42 -disable-mmx | FileCheck %s
; CHECK: jne

; widening select v6i32 and then a sub

define void @select(i1 %c, <6 x i32>* %dst.addr, <6 x i32> %src1,<6 x i32> %src2) nounwind {
entry:
	%x = select i1 %c, <6 x i32> %src1, <6 x i32> %src2
	%val = sub <6 x i32> %x, < i32 1, i32 1, i32 1, i32 1, i32 1, i32 1 >;
	store <6 x i32> %val, <6 x i32>* %dst.addr
	ret void
}
