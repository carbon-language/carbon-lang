; RUN: llc < %s -march=x86 -mattr=+sse4.2 | FileCheck %s
; CHECK: psllq $48, %xmm0
; CHECK: psrad $16, %xmm0
; CHECK: pshufd {{.*#+}} xmm0 = xmm0[1,3,2,3]

; sign extension v2i16 to v2i32

define void @convert(<2 x i32>* %dst.addr, <2 x i16> %src) nounwind {
entry:
	%signext = sext <2 x i16> %src to <2 x i32>		; <<12 x i8>> [#uses=1]
	store <2 x i32> %signext, <2 x i32>* %dst.addr
	ret void
}
