; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vcmpltps %ymm
; CHECK-NOT: vucomiss
define <8 x i32> @cmp00(<8 x float> %a, <8 x float> %b) nounwind readnone {
  %bincmp = fcmp olt <8 x float> %a, %b
  %s = sext <8 x i1> %bincmp to <8 x i32>
  ret <8 x i32> %s
}

; CHECK: vcmpltpd %ymm
; CHECK-NOT: vucomisd
define <4 x i64> @cmp01(<4 x double> %a, <4 x double> %b) nounwind readnone {
  %bincmp = fcmp olt <4 x double> %a, %b
  %s = sext <4 x i1> %bincmp to <4 x i64>
  ret <4 x i64> %s
}

declare void @scale() nounwind uwtable

; CHECK: vucomisd
define void @render() nounwind uwtable {
entry:
  br i1 undef, label %for.cond5, label %for.end52

for.cond5:
  %or.cond = and i1 undef, false
  br i1 %or.cond, label %for.body33, label %for.cond5

for.cond30:
  br i1 false, label %for.body33, label %for.cond5

for.body33:
  %tobool = fcmp une double undef, 0.000000e+00
  br i1 %tobool, label %if.then, label %for.cond30

if.then:
  call void @scale()
  br label %for.cond30

for.end52:
  ret void
}

