; RUN: llc < %s -march=x86 -mattr=+sse42 -disable-mmx | FileCheck %s
; CHECK: insertps
; CHECK: extractps

; widening shuffle v3float and then a add

define void @shuf(<3 x float>* %dst.addr, <3 x float> %src1,<3 x float> %src2) nounwind {
entry:
	%x = shufflevector <3 x float> %src1, <3 x float> %src2, <3 x i32> < i32 0, i32 4, i32 2>
	%val = fadd <3 x float> %x, %src2;
	store <3 x float> %val, <3 x float>* %dst.addr
	ret void
}
