; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s --check-prefix=ALL --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=sse4.1 | FileCheck %s --check-prefix=ALL --check-prefix=SSE41

define <4 x i32> @a(<4 x i32> %i) nounwind  {
; SSE2-LABEL: a:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movdqa {{.*#+}} xmm1 = [117,117,117,117]
; SSE2-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[1,1,3,3]
; SSE2-NEXT:    pmuludq %xmm1, %xmm0
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    pmuludq %xmm1, %xmm2
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm2[0,2,2,3]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE2-NEXT:    retq
;
; SSE41-LABEL: a:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmulld {{.*}}(%rip), %xmm0
; SSE41-NEXT:    retq
entry:
  %A = mul <4 x i32> %i, < i32 117, i32 117, i32 117, i32 117 >
  ret <4 x i32> %A
}

define <2 x i64> @b(<2 x i64> %i) nounwind  {
; ALL-LABEL: b:
; ALL:       # BB#0: # %entry
; ALL-NEXT:    movdqa {{.*#+}} xmm1 = [117,117]
; ALL-NEXT:    movdqa %xmm0, %xmm2
; ALL-NEXT:    pmuludq %xmm1, %xmm2
; ALL-NEXT:    pxor %xmm3, %xmm3
; ALL-NEXT:    pmuludq %xmm0, %xmm3
; ALL-NEXT:    psllq $32, %xmm3
; ALL-NEXT:    paddq %xmm3, %xmm2
; ALL-NEXT:    psrlq $32, %xmm0
; ALL-NEXT:    pmuludq %xmm1, %xmm0
; ALL-NEXT:    psllq $32, %xmm0
; ALL-NEXT:    paddq %xmm2, %xmm0
; ALL-NEXT:    retq
entry:
  %A = mul <2 x i64> %i, < i64 117, i64 117 >
  ret <2 x i64> %A
}

define <4 x i32> @c(<4 x i32> %i, <4 x i32> %j) nounwind  {
; SSE2-LABEL: c:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[1,1,3,3]
; SSE2-NEXT:    pmuludq %xmm1, %xmm0
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[1,1,3,3]
; SSE2-NEXT:    pmuludq %xmm2, %xmm1
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[0,2,2,3]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE2-NEXT:    retq
;
; SSE41-LABEL: c:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmulld %xmm1, %xmm0
; SSE41-NEXT:    retq
entry:
  %A = mul <4 x i32> %i, %j
  ret <4 x i32> %A
}

define <2 x i64> @d(<2 x i64> %i, <2 x i64> %j) nounwind  {
; ALL-LABEL: d:
; ALL:       # BB#0: # %entry
; ALL-NEXT:    movdqa %xmm0, %xmm2
; ALL-NEXT:    pmuludq %xmm1, %xmm2
; ALL-NEXT:    movdqa %xmm1, %xmm3
; ALL-NEXT:    psrlq $32, %xmm3
; ALL-NEXT:    pmuludq %xmm0, %xmm3
; ALL-NEXT:    psllq $32, %xmm3
; ALL-NEXT:    paddq %xmm3, %xmm2
; ALL-NEXT:    psrlq $32, %xmm0
; ALL-NEXT:    pmuludq %xmm1, %xmm0
; ALL-NEXT:    psllq $32, %xmm0
; ALL-NEXT:    paddq %xmm2, %xmm0
; ALL-NEXT:    retq
entry:
  %A = mul <2 x i64> %i, %j
  ret <2 x i64> %A
}

declare void @foo()

define <4 x i32> @e(<4 x i32> %i, <4 x i32> %j) nounwind  {
; SSE2-LABEL: e:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    subq $40, %rsp
; SSE2-NEXT:    movaps %xmm1, {{[0-9]+}}(%rsp) # 16-byte Spill
; SSE2-NEXT:    movaps %xmm0, (%rsp) # 16-byte Spill
; SSE2-NEXT:    callq foo
; SSE2-NEXT:    movdqa (%rsp), %xmm0 # 16-byte Reload
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[1,1,3,3]
; SSE2-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm2 # 16-byte Reload
; SSE2-NEXT:    pmuludq %xmm2, %xmm0
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    pshufd {{.*#+}} xmm2 = xmm2[1,1,3,3]
; SSE2-NEXT:    pmuludq %xmm1, %xmm2
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm2[0,2,2,3]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE2-NEXT:    addq $40, %rsp
; SSE2-NEXT:    retq
;
; SSE41-LABEL: e:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    subq $40, %rsp
; SSE41-NEXT:    movaps %xmm1, {{[0-9]+}}(%rsp) # 16-byte Spill
; SSE41-NEXT:    movaps %xmm0, (%rsp) # 16-byte Spill
; SSE41-NEXT:    callq foo
; SSE41-NEXT:    movdqa (%rsp), %xmm0 # 16-byte Reload
; SSE41-NEXT:    pmulld {{[0-9]+}}(%rsp), %xmm0 # 16-byte Folded Reload
; SSE41-NEXT:    addq $40, %rsp
; SSE41-NEXT:    retq
entry:
  ; Use a call to force spills.
  call void @foo()
  %A = mul <4 x i32> %i, %j
  ret <4 x i32> %A
}

define <2 x i64> @f(<2 x i64> %i, <2 x i64> %j) nounwind  {
; ALL-LABEL: f:
; ALL:       # BB#0: # %entry
; ALL-NEXT:    subq $40, %rsp
; ALL-NEXT:    movaps %xmm1, {{[0-9]+}}(%rsp) # 16-byte Spill
; ALL-NEXT:    movaps %xmm0, (%rsp) # 16-byte Spill
; ALL-NEXT:    callq foo
; ALL-NEXT:    movdqa (%rsp), %xmm0 # 16-byte Reload
; ALL-NEXT:    movdqa %xmm0, %xmm2
; ALL-NEXT:    movdqa {{[0-9]+}}(%rsp), %xmm3 # 16-byte Reload
; ALL-NEXT:    pmuludq %xmm3, %xmm2
; ALL-NEXT:    movdqa %xmm3, %xmm1
; ALL-NEXT:    psrlq $32, %xmm1
; ALL-NEXT:    pmuludq %xmm0, %xmm1
; ALL-NEXT:    psllq $32, %xmm1
; ALL-NEXT:    paddq %xmm1, %xmm2
; ALL-NEXT:    psrlq $32, %xmm0
; ALL-NEXT:    pmuludq %xmm3, %xmm0
; ALL-NEXT:    psllq $32, %xmm0
; ALL-NEXT:    paddq %xmm2, %xmm0
; ALL-NEXT:    addq $40, %rsp
; ALL-NEXT:    retq
entry:
  ; Use a call to force spills.
  call void @foo()
  %A = mul <2 x i64> %i, %j
  ret <2 x i64> %A
}
