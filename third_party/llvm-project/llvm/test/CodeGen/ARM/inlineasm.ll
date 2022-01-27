; RUN: llc -mtriple=armv8-eabi -mattr=+neon %s -o - | FileCheck %s

define i32 @test1(i32 %tmp54) {
	%tmp56 = tail call i32 asm "uxtb16 $0,$1", "=r,r"( i32 %tmp54 )		; <i32> [#uses=1]
	ret i32 %tmp56
}

define void @test2() {
	tail call void asm sideeffect "/* number: ${0:c} */", "i"( i32 1 )
	ret void
}

define float @t-constraint-int(i32 %i) {
	; CHECK-LABEL: t-constraint-int
	; CHECK: vcvt.f32.s32 {{s[0-9]+}}, {{s[0-9]+}}
	%ret = call float asm "vcvt.f32.s32 $0, $1\0A", "=t,t"(i32 %i)
	ret float %ret
}

define <2 x i32> @t-constraint-int-vector-64bit(<2 x float> %x) {
entry:
	; CHECK-LABEL: t-constraint-int-vector-64bit
	; CHECK: vcvt.s32.f32 {{d[0-9]+}}, {{d[0-9]+}}
  %0 = tail call <2 x i32> asm "vcvt.s32.f32 $0, $1", "=t,t"(<2 x float> %x)
  ret <2 x i32> %0
}

define <4 x i32> @t-constraint-int-vector-128bit(<4 x float> %x) {
entry:
	; CHECK-LABEL: t-constraint-int-vector-128bit
	; CHECK: vcvt.s32.f32 {{q[0-7]}}, {{q[0-7]}}
  %0 = tail call <4 x i32> asm "vcvt.s32.f32 $0, $1", "=t,t"(<4 x float> %x)
  ret <4 x i32> %0
}

define <2 x float> @t-constraint-float-vector-64bit(<2 x float> %a, <2 x float> %b) {
entry:
	; CHECK-LABEL: t-constraint-float-vector-64bit
	; CHECK: vadd.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
	%0 = tail call <2 x float> asm "vadd.f32 $0, $1, $2", "=t,t,t"(<2 x float> %a, <2 x float> %b)
	ret <2 x float> %0
}

define <4 x float> @t-constraint-float-vector-128bit(<4 x float> %a, <4 x float> %b) {
entry:
	; CHECK-LABEL: t-constraint-float-vector-128bit
	; CHECK: vadd.f32 q{{[0-7]}}, q{{[0-7]}}, q{{[0-7]}}
	%0 = tail call <4 x float> asm "vadd.f32 $0, $1, $2", "=t,t,t"(<4 x float> %a, <4 x float> %b)
	ret <4 x float> %0
}

define i32 @even-GPR-constraint() {
entry:
	; CHECK-LABEL: even-GPR-constraint
	; CHECK: add [[REG:r1*[0, 2, 4, 6, 8]]], [[REG]], #1
	; CHECK: add [[REG:r1*[0, 2, 4, 6, 8]]], [[REG]], #2
	; CHECK: add [[REG:r1*[0, 2, 4, 6, 8]]], [[REG]], #3
	; CHECK: add [[REG:r1*[0, 2, 4, 6, 8]]], [[REG]], #4
	%0 = tail call { i32, i32, i32, i32 } asm "add $0, #1\0Aadd $1, #2\0Aadd $2, #3\0Aadd $3, #4\0A", "=^Te,=^Te,=^Te,=^Te,0,1,2,3"(i32 0, i32 0, i32 0, i32 0)
	%asmresult = extractvalue { i32, i32, i32, i32 } %0, 0
	ret i32 %asmresult
}

define i32 @odd-GPR-constraint() {
entry:
	; CHECK-LABEL: odd-GPR-constraint
	; CHECK: add [[REG:r1*[1, 3, 5, 7, 9]]], [[REG]], #1
	; CHECK: add [[REG:r1*[1, 3, 5, 7, 9]]], [[REG]], #2
	; CHECK: add [[REG:r1*[1, 3, 5, 7, 9]]], [[REG]], #3
	; CHECK: add [[REG:r1*[1, 3, 5, 7, 9]]], [[REG]], #4
	%0 = tail call { i32, i32, i32, i32 } asm "add $0, #1\0Aadd $1, #2\0Aadd $2, #3\0Aadd $3, #4\0A", "=^To,=^To,=^To,=^To,0,1,2,3"(i32 0, i32 0, i32 0, i32 0)
	%asmresult = extractvalue { i32, i32, i32, i32 } %0, 0
	ret i32 %asmresult
}
