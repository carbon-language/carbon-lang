; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

declare void @scale() nounwind uwtable

; CHECK: vucomisd .LCPI
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

