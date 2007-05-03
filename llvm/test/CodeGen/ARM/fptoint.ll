; RUN: llvm-as < %s | llc -march=arm -mattr=+v6,+vfp2 > %t
; RUN: not grep fmrs %t

@i = weak global i32 0		; <i32*> [#uses=2]
@u = weak global i32 0		; <i32*> [#uses=2]

define void @foo5(float %x) {
entry:
	%tmp1 = fptosi float %x to i32		; <i32> [#uses=1]
	store i32 %tmp1, i32* @i
	ret void
}

define void @foo6(float %x) {
entry:
	%tmp1 = fptoui float %x to i32		; <i32> [#uses=1]
	store i32 %tmp1, i32* @u
	ret void
}

define void @foo7(double %x) {
entry:
	%tmp1 = fptosi double %x to i32		; <i32> [#uses=1]
	store i32 %tmp1, i32* @i
	ret void
}

define void @foo8(double %x) {
entry:
	%tmp1 = fptoui double %x to i32		; <i32> [#uses=1]
	store i32 %tmp1, i32* @u
	ret void
}
