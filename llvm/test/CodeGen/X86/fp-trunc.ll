; RUN: llc < %s -march=x86 -mattr=+sse2,-avx | FileCheck %s

define <1 x float> @test1(<1 x double> %x) nounwind {
; CHECK: cvtsd2ss
; CHECK: ret
  %y = fptrunc <1 x double> %x to <1 x float>
  ret <1 x float> %y
}


define <2 x float> @test2(<2 x double> %x) nounwind {
; FIXME: It would be nice if this compiled down to a cvtpd2ps
; CHECK: cvtsd2ss
; CHECK: cvtsd2ss
; CHECK: ret
  %y = fptrunc <2 x double> %x to <2 x float>
  ret <2 x float> %y
}

define <8 x float> @test3(<8 x double> %x) nounwind {
; FIXME: It would be nice if this compiled down to a series of cvtpd2ps
; CHECK: cvtsd2ss
; CHECK: cvtsd2ss
; CHECK: cvtsd2ss
; CHECK: cvtsd2ss
; CHECK: cvtsd2ss
; CHECK: cvtsd2ss
; CHECK: cvtsd2ss
; CHECK: cvtsd2ss
; CHECK: ret
  %y = fptrunc <8 x double> %x to <8 x float>
  ret <8 x float> %y
}


