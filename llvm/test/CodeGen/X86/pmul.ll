; RUN: llc < %s -march=x86 -mattr=sse4.1 -mcpu=nehalem -stack-alignment=16 | FileCheck %s

define <4 x i32> @a(<4 x i32> %i) nounwind  {
; CHECK-LABEL: a:
; CHECK:         pmulld
; CHECK-NEXT:    retl
entry:
  %A = mul <4 x i32> %i, < i32 117, i32 117, i32 117, i32 117 >
  ret <4 x i32> %A
}

define <2 x i64> @b(<2 x i64> %i) nounwind  {
; CHECK-LABEL: b:
; CHECK:         pmuludq
; CHECK:         pmuludq
; CHECK:         pmuludq
entry:
  %A = mul <2 x i64> %i, < i64 117, i64 117 >
  ret <2 x i64> %A
}

define <4 x i32> @c(<4 x i32> %i, <4 x i32> %j) nounwind  {
; CHECK-LABEL: c:
; CHECK:         pmulld
; CHECK-NEXT:    retl
entry:
  %A = mul <4 x i32> %i, %j
  ret <4 x i32> %A
}

define <2 x i64> @d(<2 x i64> %i, <2 x i64> %j) nounwind  {
; CHECK-LABEL: d:
; CHECK:         pmuludq
; CHECK:         pmuludq
; CHECK:         pmuludq
entry:
  %A = mul <2 x i64> %i, %j
  ret <2 x i64> %A
}

declare void @foo()

define <4 x i32> @e(<4 x i32> %i, <4 x i32> %j) nounwind  {
; CHECK-LABEL: e:
; CHECK:         pmulld {{[0-9]+}}(%esp), %xmm
; CHECK-NEXT:    addl ${{[0-9]+}}, %esp
; CHECK-NEXT:    retl
entry:
  ; Use a call to force spills.
  call void @foo()
  %A = mul <4 x i32> %i, %j
  ret <4 x i32> %A
}

define <2 x i64> @f(<2 x i64> %i, <2 x i64> %j) nounwind  {
; CHECK-LABEL: f:
; CHECK:         pmuludq
; CHECK:         pmuludq
; CHECK:         pmuludq
entry:
  ; Use a call to force spills.
  call void @foo()
  %A = mul <2 x i64> %i, %j
  ret <2 x i64> %A
}
