; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s --check-prefix=ALL --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=sse4.1 | FileCheck %s --check-prefix=ALL --check-prefix=SSE41

define <4 x i32> @a(<4 x i32> %i) nounwind  {
; SSE2-LABEL: a:
; SSE2:         movdqa {{.*}}, %[[X1:xmm[0-9]+]]
; SSE2-NEXT:    pshufd {{.*}} # [[X2:xmm[0-9]+]] = xmm0[1,0,3,0]
; SSE2-NEXT:    pmuludq %[[X1]], %xmm0
; SSE2-NEXT:    pmuludq %[[X1]], %[[X2]]
; SSE2-NEXT:    shufps {{.*}} # xmm0 = xmm0[0,2],[[X2]][0,2]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,1,3]
; SSE2-NEXT:    retq
;
; SSE41-LABEL: a:
; SSE41:         pmulld
; SSE41-NEXT:    retq
entry:
  %A = mul <4 x i32> %i, < i32 117, i32 117, i32 117, i32 117 >
  ret <4 x i32> %A
}

define <2 x i64> @b(<2 x i64> %i) nounwind  {
; ALL-LABEL: b:
; ALL:         pmuludq
; ALL:         pmuludq
; ALL:         pmuludq
entry:
  %A = mul <2 x i64> %i, < i64 117, i64 117 >
  ret <2 x i64> %A
}

define <4 x i32> @c(<4 x i32> %i, <4 x i32> %j) nounwind  {
; SSE2-LABEL: c:
; SSE2:         pshufd {{.*}} # [[X2:xmm[0-9]+]] = xmm0[1,0,3,0]
; SSE2-NEXT:    pmuludq %xmm1, %xmm0
; SSE2-NEXT:    pshufd {{.*}} # xmm1 = xmm1[1,0,3,0]
; SSE2-NEXT:    pmuludq %[[X2]], %xmm1
; SSE2-NEXT:    shufps {{.*}} # xmm0 = xmm0[0,2],xmm1[0,2]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,1,3]
; SSE2-NEXT:    retq
;
; SSE41-LABEL: c:
; SSE41:         pmulld
; SSE41-NEXT:    retq
entry:
  %A = mul <4 x i32> %i, %j
  ret <4 x i32> %A
}

define <2 x i64> @d(<2 x i64> %i, <2 x i64> %j) nounwind  {
; ALL-LABEL: d:
; ALL:         pmuludq
; ALL:         pmuludq
; ALL:         pmuludq
entry:
  %A = mul <2 x i64> %i, %j
  ret <2 x i64> %A
}

declare void @foo()

define <4 x i32> @e(<4 x i32> %i, <4 x i32> %j) nounwind  {
; SSE2-LABEL: e:
; SSE2:         movdqa {{[0-9]*}}(%rsp), %[[X1:xmm[0-9]+]]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = [[X2]][1,0,3,0]
; SSE2-NEXT:    movdqa {{[0-9]*}}(%rsp), %[[X2:xmm[0-9]+]]
; SSE2-NEXT:    pmuludq %[[X2]], %[[X1]]
; SSE2-NEXT:    pshufd {{.*}} # [[X2]] = [[X2]][1,0,3,0]
; SSE2-NEXT:    pmuludq %xmm0, %[[X2]]
; SSE2-NEXT:    shufps {{.*}} # [[X1]] = [[X1]][0,2],[[X2]][0,2]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = [[X1]][0,2,1,3]
; SSE2-NEXT:    addq ${{[0-9]+}}, %rsp
; SSE2-NEXT:    retq
;
; SSE41-LABEL: e:
; SSE41:         pmulld {{[0-9]+}}(%rsp), %xmm
; SSE41-NEXT:    addq ${{[0-9]+}}, %rsp
; SSE41-NEXT:    retq
entry:
  ; Use a call to force spills.
  call void @foo()
  %A = mul <4 x i32> %i, %j
  ret <4 x i32> %A
}

define <2 x i64> @f(<2 x i64> %i, <2 x i64> %j) nounwind  {
; ALL-LABEL: f:
; ALL:         pmuludq
; ALL:         pmuludq
; ALL:         pmuludq
entry:
  ; Use a call to force spills.
  call void @foo()
  %A = mul <2 x i64> %i, %j
  ret <2 x i64> %A
}
