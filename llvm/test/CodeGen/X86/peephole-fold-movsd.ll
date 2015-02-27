; RUN: llc -mtriple=x86_64-pc-linux < %s | FileCheck %s
;
; Check that x86's peephole optimization doesn't fold a 64-bit load (movsd) into
; addpd.
; rdar://problem/18236850

%struct.S1 = type { double, double }

@g = common global %struct.S1 zeroinitializer, align 8

declare void @foo3(%struct.S1*)

; CHECK: movsd {{[0-9]*}}(%rsp), [[R0:%xmm[0-9]+]]
; CHECK: addpd [[R0]], %xmm{{[0-9]+}}

define void @foo1(double %a.coerce0, double %a.coerce1, double %b.coerce0, double %b.coerce1) {
  %1 = alloca <2 x double>, align 16
  %tmpcast = bitcast <2 x double>* %1 to %struct.S1*
  call void @foo3(%struct.S1* %tmpcast) #2
  %p2 = getelementptr inbounds %struct.S1, %struct.S1* %tmpcast, i64 0, i32 0
  %2 = load double* %p2, align 16
  %p3 = getelementptr inbounds %struct.S1, %struct.S1* %tmpcast, i64 0, i32 1
  %3 = load double* %p3, align 8
  %4 = insertelement <2 x double> undef, double %2, i32 0
  %5 = insertelement <2 x double> %4, double 0.000000e+00, i32 1
  %6 = insertelement <2 x double> undef, double %3, i32 1
  %7 = insertelement <2 x double> %6, double 1.000000e+00, i32 0
  %8 = fadd <2 x double> %5, %7
  store <2 x double> %8, <2 x double>* bitcast (%struct.S1* @g to <2 x double>*), align 16
  ret void
}
