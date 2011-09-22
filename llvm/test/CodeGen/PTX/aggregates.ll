; RUN: llc < %s -march=ptx32 -mattr=sm20 | FileCheck %s
; XFAIL: *

%complex = type { float, float }

define ptx_device %complex @complex_add(%complex %a, %complex %b) {
entry:
; CHECK:      ld.param.f32	r[[R0:[0-9]+]], [__param_1];
; CHECK-NEXT:	ld.param.f32	r[[R2:[0-9]+]], [__param_3];
; CHECK-NEXT:	ld.param.f32	r[[R1:[0-9]+]], [__param_2];
; CHECK-NEXT:	ld.param.f32	r[[R3:[0-9]+]], [__param_4];
; CHECK-NEXT:	add.rn.f32	r[[R0]], r[[R0]], r[[R2]];
; CHECK-NEXT:	add.rn.f32	r[[R1]], r[[R1]], r[[R3]];
; CHECK-NEXT:	ret;
  %a.real = extractvalue %complex %a, 0
  %a.imag = extractvalue %complex %a, 1
  %b.real = extractvalue %complex %b, 0
  %b.imag = extractvalue %complex %b, 1
  %ret.real = fadd float %a.real, %b.real
  %ret.imag = fadd float %a.imag, %b.imag
  %ret.0 = insertvalue %complex undef, float %ret.real, 0
  %ret.1 = insertvalue %complex %ret.0, float %ret.imag, 1
  ret %complex %ret.1
}
