; RUN: llc -asm-verbose=false -mtriple=mipsel-linux-gnu -relocation-model=pic < %s | FileCheck %s -check-prefix=CHECK-LIBCALL

; CHECK-LIBCALL-LABEL: test_fadd:
; CHECK-LIBCALL: %call16(__gnu_h2f_ieee)
; CHECK-LIBCALL: %call16(__gnu_h2f_ieee)
; CHECK-LIBCALL-DAG: add.s
; CHECK-LIBCALL-DAG: %call16(__gnu_f2h_ieee)
define void @test_fadd(half* %p, half* %q) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %r = fadd half %a, %b
  store half %r, half* %p
  ret void
}

; CHECK-LIBCALL-LABEL: test_fpext_float:
; CHECK-LIBCALL: %call16(__gnu_h2f_ieee)
define float @test_fpext_float(half* %p) {
  %a = load half, half* %p, align 2
  %r = fpext half %a to float
  ret float %r
}

; CHECK-LIBCALL-LABEL: test_fpext_double:
; CHECK-LIBCALL: %call16(__gnu_h2f_ieee)
; CHECK-LIBCALL: cvt.d.s
define double @test_fpext_double(half* %p) {
  %a = load half, half* %p, align 2
  %r = fpext half %a to double
  ret double %r
}

; CHECK-LIBCALL-LABEL: test_fptrunc_float:
; CHECK-LIBCALL: %call16(__gnu_f2h_ieee)
define void @test_fptrunc_float(float %f, half* %p) #0 {
  %a = fptrunc float %f to half
  store half %a, half* %p
  ret void
}

; CHECK-LIBCALL-LABEL: test_fptrunc_double:
; CHECK-LIBCALL: %call16(__truncdfhf2)
define void @test_fptrunc_double(double %d, half* %p) #0 {
  %a = fptrunc double %d to half
  store half %a, half* %p
  ret void
}

; CHECK-LIBCALL-LABEL: test_vec_fpext_float:
; CHECK-LIBCALL: %call16(__gnu_h2f_ieee)
; CHECK-LIBCALL: %call16(__gnu_h2f_ieee)
; CHECK-LIBCALL: %call16(__gnu_h2f_ieee)
; CHECK-LIBCALL: %call16(__gnu_h2f_ieee)
define <4 x float> @test_vec_fpext_float(<4 x half>* %p) #0 {
  %a = load <4 x half>, <4 x half>* %p, align 8
  %b = fpext <4 x half> %a to <4 x float>
  ret <4 x float> %b
}

; This test is not robust against variations in instruction scheduling.
; See the discussion in http://reviews.llvm.org/D8804
; CHECK-LIBCALL-LABEL: test_vec_fpext_double:
; CHECK-LIBCALL: %call16(__gnu_h2f_ieee)
; CHECK-LIBCALL: %call16(__gnu_h2f_ieee)
; CHECK-LIBCALL: %call16(__gnu_h2f_ieee)
; CHECK-LIBCALL: cvt.d.s
; CHECK-LIBCALL: cvt.d.s
; CHECK-LIBCALL: cvt.d.s
; CHECK-LIBCALL: %call16(__gnu_h2f_ieee)
; CHECK-LIBCALL: cvt.d.s
define <4 x double> @test_vec_fpext_double(<4 x half>* %p) #0 {
  %a = load <4 x half>, <4 x half>* %p, align 8
  %b = fpext <4 x half> %a to <4 x double>
  ret <4 x double> %b
}

; CHECK-LIBCALL-LABEL: test_vec_fptrunc_float:
; CHECK-LIBCALL: %call16(__gnu_f2h_ieee)
; CHECK-LIBCALL: %call16(__gnu_f2h_ieee)
; CHECK-LIBCALL: %call16(__gnu_f2h_ieee)
; CHECK-LIBCALL: %call16(__gnu_f2h_ieee)
define void @test_vec_fptrunc_float(<4 x float> %a, <4 x half>* %p) #0 {
  %b = fptrunc <4 x float> %a to <4 x half>
  store <4 x half> %b, <4 x half>* %p, align 8
  ret void
}

; CHECK-LIBCALL-LABEL: test_vec_fptrunc_double:
; CHECK-LIBCALL: %call16(__truncdfhf2)
; CHECK-LIBCALL: %call16(__truncdfhf2)
; CHECK-LIBCALL: %call16(__truncdfhf2)
; CHECK-LIBCALL: %call16(__truncdfhf2)
define void @test_vec_fptrunc_double(<4 x double> %a, <4 x half>* %p) #0 {
  %b = fptrunc <4 x double> %a to <4 x half>
  store <4 x half> %b, <4 x half>* %p, align 8
  ret void
}

