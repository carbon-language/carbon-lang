; RUN: llc -mcpu=pwr7 -mattr=+vsx < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define double @foo1(double %a) nounwind {
entry:
  call void asm sideeffect "", "~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f18},~{f19},~{f20},~{f21},~{f22},~{f23},~{f24},~{f25},~{f26},~{f27},~{f28},~{f29},~{f30},~{f31}"() nounwind
  br label %return

; CHECK: @foo1
; CHECK: xxlor [[R1:[0-9]+]], 1, 1
; CHECK: xxlor 1, [[R1]], [[R1]]
; CHECK: blr

return:                                           ; preds = %entry
  ret double %a
}

define double @foo2(double %a) nounwind {
entry:
  %b = fadd double %a, %a
  call void asm sideeffect "", "~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f18},~{f19},~{f20},~{f21},~{f22},~{f23},~{f24},~{f25},~{f26},~{f27},~{f28},~{f29},~{f30},~{f31}"() nounwind
  br label %return

; CHECK: @foo2
; CHECK: {{xxlor|xsadddp}} [[R1:[0-9]+]], 1, 1
; CHECK: {{xxlor|xsadddp}} 1, [[R1]], [[R1]]
; CHECK: blr

return:                                           ; preds = %entry
  ret double %b
}

define double @foo3(double %a) nounwind {
entry:
  call void asm sideeffect "", "~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f18},~{f19},~{f20},~{f21},~{f22},~{f23},~{f24},~{f25},~{f26},~{f27},~{f28},~{f29},~{f30},~{f31},~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"() nounwind
  br label %return

; CHECK: @foo3
; CHECK: stxsdx 1,
; CHECK: lxsdx [[R1:[0-9]+]],
; CHECK: xsadddp 1, [[R1]], [[R1]]
; CHECK: blr

return:                                           ; preds = %entry
  %b = fadd double %a, %a
  ret double %b
}

