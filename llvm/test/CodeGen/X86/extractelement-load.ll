; RUN: llc < %s -march=x86 -mattr=+sse2 -mcpu=yonah | FileCheck %s
; RUN: llc < %s -march=x86-64 -mattr=+sse2 -mcpu=core2 | FileCheck %s
; RUN: llc < %s -march=x86-64 -mattr=+avx -mcpu=btver2 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @t(<2 x i64>* %val) nounwind  {
; CHECK-LABEL: t:
; CHECK-NOT: movd
; CHECK: movl 8(
; CHECK-NEXT: ret
	%tmp2 = load <2 x i64>, <2 x i64>* %val, align 16		; <<2 x i64>> [#uses=1]
	%tmp3 = bitcast <2 x i64> %tmp2 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp4 = extractelement <4 x i32> %tmp3, i32 2		; <i32> [#uses=1]
	ret i32 %tmp4
}

; Case where extractelement of load ends up as undef.
; (Making sure this doesn't crash.)
define i32 @t2(<8 x i32>* %xp) {
; CHECK-LABEL: t2:
; CHECK: ret
  %x = load <8 x i32>, <8 x i32>* %xp
  %Shuff68 = shufflevector <8 x i32> %x, <8 x i32> undef, <8 x i32> <i32
undef, i32 7, i32 9, i32 undef, i32 13, i32 15, i32 1, i32 3>
  %y = extractelement <8 x i32> %Shuff68, i32 0
  ret i32 %y
}

; This case could easily end up inf-looping in the DAG combiner due to an
; low alignment load of the vector which prevents us from reliably forming a
; narrow load.

; The expected codegen is identical for the AVX case except
; load/store instructions will have a leading 'v', so we don't
; need to special-case the checks.

define void @t3() {
; CHECK-LABEL: t3:
; CHECK: movupd
; CHECK: movhpd

bb:
  %tmp13 = load <2 x double>, <2 x double>* undef, align 1
  %.sroa.3.24.vec.extract = extractelement <2 x double> %tmp13, i32 1
  store double %.sroa.3.24.vec.extract, double* undef, align 8
  unreachable
}

; Case where a load is unary shuffled, then bitcast (to a type with the same
; number of elements) before extractelement.
; This is testing for an assertion - the extraction was assuming that the undef
; second shuffle operand was a post-bitcast type instead of a pre-bitcast type.
define i64 @t4(<2 x double>* %a) {
; CHECK-LABEL: t4:
; CHECK: mov
; CHECK: ret
  %b = load <2 x double>, <2 x double>* %a, align 16
  %c = shufflevector <2 x double> %b, <2 x double> %b, <2 x i32> <i32 1, i32 0>
  %d = bitcast <2 x double> %c to <2 x i64>
  %e = extractelement <2 x i64> %d, i32 1
  ret i64 %e
}

