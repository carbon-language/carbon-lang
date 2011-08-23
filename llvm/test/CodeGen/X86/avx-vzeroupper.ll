; RUN: llc < %s -x86-use-vzeroupper -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

define <4 x float> @do_sse_local(<4 x float> %a) nounwind uwtable readnone ssp {
entry:
  %add.i = fadd <4 x float> %a, %a
  ret <4 x float> %add.i
}

; CHECK: _test00
define <4 x float> @test00(<4 x float> %a, <4 x float> %b) nounwind uwtable ssp {
entry:
  %add.i = fadd <4 x float> %a, %b
  ; CHECK: vzeroupper
  ; CHECK-NEXT: callq _do_sse
  %call3 = tail call <4 x float> @do_sse(<4 x float> %add.i) nounwind
  %sub.i = fsub <4 x float> %call3, %add.i
  ; CHECK-NOT: vzeroupper
  ; CHECK: callq _do_sse_local
  %call8 = tail call <4 x float> @do_sse_local(<4 x float> %sub.i)
  ; CHECK: vzeroupper
  ; CHECK-NEXT: jmp _do_sse
  %call10 = tail call <4 x float> @do_sse(<4 x float> %call8) nounwind
  ret <4 x float> %call10
}

declare <4 x float> @do_sse(<4 x float>)
