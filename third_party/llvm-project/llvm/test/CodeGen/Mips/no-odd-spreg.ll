; RUN: llc -march=mipsel -mcpu=mips32 < %s \
; RUN:   | FileCheck %s -check-prefixes=ALL,ODDSPREG,ODDSPREG-NO-EMIT
; RUN: llc -march=mipsel -mcpu=mips32 -mattr=+nooddspreg < %s \
; RUN:   | FileCheck %s -check-prefixes=ALL,NOODDSPREG
; RUN: llc -march=mipsel -mcpu=mips32r6 -mattr=fp64 < %s \
; RUN:   | FileCheck %s -check-prefixes=ALL,ODDSPREG,ODDSPREG-NO-EMIT
; RUN: llc -march=mipsel -mcpu=mips32r6 -mattr=fp64,+nooddspreg < %s \
; RUN:   | FileCheck %s -check-prefixes=ALL,NOODDSPREG
; RUN: llc -march=mipsel -mcpu=mips32r6 -mattr=fpxx,-nooddspreg < %s \
; RUN:   | FileCheck %s -check-prefixes=ALL,ODDSPREG,ODDSPREG-EMIT

; We don't emit a directive unless we need to. This is to support versions of
; GAS which do not support the directive.
; ODDSPREG-EMIT:        .module oddspreg
; ODDSPREG-NO-EMIT-NOT: .module oddspreg
; NOODDSPREG:           .module nooddspreg

define float @two_floats(float %a) {
entry:
  ; Clobber all except $f12 and $f13
  ;
  ; The intention is that if odd single precision registers are permitted, the
  ; allocator will choose $f12 and $f13 to avoid the spill/reload.
  ;
  ; On the other hand, if odd single precision registers are not permitted, it
  ; will be forced to spill/reload either %a or %0.
  ;
  ; This is affected by scheduling as the new machine scheduler schedules the
  ; two adds together, avoiding the spill+reload.

  %0 = fadd float %a, 1.0
  call void asm "# Clobber", "~{$f0},~{$f1},~{$f2},~{$f3},~{$f4},~{$f5},~{$f6},~{$f7},~{$f8},~{$f9},~{$f10},~{$f11},~{$f14},~{$f15},~{$f16},~{$f17},~{$f18},~{$f19},~{$f20},~{$f21},~{$f22},~{$f23},~{$f24},~{$f25},~{$f26},~{$f27},~{$f28},~{$f29},~{$f30},~{$f31}"()
  %1 = fadd float %a, %0
  ret float %1
}

; ALL-LABEL:  two_floats:
; ODDSPREG:       add.s $f[[RES:[0-9]]], $f12, ${{f[0-9]+}}
; ODDSPREG:       add.s ${{f[0-9]+}}, $f12, $f[[RES]]
; ODDSPREG-NOT:   swc1
; ODDSPREG-NOT:   lwc1

; NOODDSPREG:     add.s $[[T0:f[0-9]*[02468]]], $f12, ${{f[0-9]+}}
; NOODDSPREG-NOT: swc1 $[[T0]],
; NOODDSPREG-NOT: lwc1 $[[T1:f[0-9]*[02468]]],
; NOODDSPREG:     add.s ${{f[0-9]+}}, $f12, $[[T0]]

define double @two_doubles(double %a) {
entry:
  ; Clobber all except $f12 and $f13
  ;
  ; -mno-odd-sp-reg doesn't need to affect double precision values so both cases
  ; use $f12 and $f13.

  %0 = fadd double %a, 1.0
  call void asm "# Clobber", "~{$f0},~{$f1},~{$f2},~{$f3},~{$f4},~{$f5},~{$f6},~{$f7},~{$f8},~{$f9},~{$f10},~{$f11},~{$f14},~{$f15},~{$f16},~{$f17},~{$f18},~{$f19},~{$f20},~{$f21},~{$f22},~{$f23},~{$f24},~{$f25},~{$f26},~{$f27},~{$f28},~{$f29},~{$f30},~{$f31}"()
  %1 = fadd double %a, %0
  ret double %1
}

; ALL-LABEL: two_doubles:
; ALL-DAG:        add.d $[[T0:f[0-9]+]], $f12, ${{f[0-9]+}}
; ALL-DAG:        add.d ${{f[0-9]+}}, $f12, $[[T0]]


; INVALID: -mattr=+nooddspreg is not currently permitted for a 32-bit FPU register file (FR=0 mode).
