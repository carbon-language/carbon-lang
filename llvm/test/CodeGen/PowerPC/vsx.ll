; RUN: llc -mcpu=pwr7 -mattr=+vsx < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define double @test1(double %a, double %b) {
entry:
  %v = fmul double %a, %b
  ret double %v

; CHECK-LABEL: @test1
; CHECK: xsmuldp 1, 1, 2
; CHECK: blr
}

define double @test2(double %a, double %b) {
entry:
  %v = fdiv double %a, %b
  ret double %v

; CHECK-LABEL: @test2
; CHECK: xsdivdp 1, 1, 2
; CHECK: blr
}

define double @test3(double %a, double %b) {
entry:
  %v = fadd double %a, %b
  ret double %v

; CHECK-LABEL: @test3
; CHECK: xsadddp 1, 1, 2
; CHECK: blr
}

define <2 x double> @test4(<2 x double> %a, <2 x double> %b) {
entry:
  %v = fadd <2 x double> %a, %b
  ret <2 x double> %v

; CHECK-LABEL: @test4
; CHECK: xvadddp 34, 34, 35
; CHECK: blr
}

define <4 x i32> @test5(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = xor <4 x i32> %a, %b
  ret <4 x i32> %v

; CHECK-LABEL: @test5
; CHECK: xxlxor 34, 34, 35
; CHECK: blr
}

define <8 x i16> @test6(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = xor <8 x i16> %a, %b
  ret <8 x i16> %v

; CHECK-LABEL: @test6
; CHECK: xxlxor 34, 34, 35
; CHECK: blr
}

define <16 x i8> @test7(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = xor <16 x i8> %a, %b
  ret <16 x i8> %v

; CHECK-LABEL: @test7
; CHECK: xxlxor 34, 34, 35
; CHECK: blr
}

define <4 x i32> @test8(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = or <4 x i32> %a, %b
  ret <4 x i32> %v

; CHECK-LABEL: @test8
; CHECK: xxlor 34, 34, 35
; CHECK: blr
}

define <8 x i16> @test9(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = or <8 x i16> %a, %b
  ret <8 x i16> %v

; CHECK-LABEL: @test9
; CHECK: xxlor 34, 34, 35
; CHECK: blr
}

define <16 x i8> @test10(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = or <16 x i8> %a, %b
  ret <16 x i8> %v

; CHECK-LABEL: @test10
; CHECK: xxlor 34, 34, 35
; CHECK: blr
}

define <4 x i32> @test11(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = and <4 x i32> %a, %b
  ret <4 x i32> %v

; CHECK-LABEL: @test11
; CHECK: xxland 34, 34, 35
; CHECK: blr
}

define <8 x i16> @test12(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = and <8 x i16> %a, %b
  ret <8 x i16> %v

; CHECK-LABEL: @test12
; CHECK: xxland 34, 34, 35
; CHECK: blr
}

define <16 x i8> @test13(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = and <16 x i8> %a, %b
  ret <16 x i8> %v

; CHECK-LABEL: @test13
; CHECK: xxland 34, 34, 35
; CHECK: blr
}

define <4 x i32> @test14(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = or <4 x i32> %a, %b
  %w = xor <4 x i32> %v, <i32 -1, i32 -1, i32 -1, i32 -1>
  ret <4 x i32> %w

; CHECK-LABEL: @test14
; CHECK: xxlnor 34, 34, 35
; CHECK: blr
}

define <8 x i16> @test15(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = or <8 x i16> %a, %b
  %w = xor <8 x i16> %v, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  ret <8 x i16> %w

; CHECK-LABEL: @test15
; CHECK: xxlnor 34, 34, 35
; CHECK: blr
}

define <16 x i8> @test16(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = or <16 x i8> %a, %b
  %w = xor <16 x i8> %v, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  ret <16 x i8> %w

; CHECK-LABEL: @test16
; CHECK: xxlnor 34, 34, 35
; CHECK: blr
}

define <4 x i32> @test17(<4 x i32> %a, <4 x i32> %b) {
entry:
  %w = xor <4 x i32> %b, <i32 -1, i32 -1, i32 -1, i32 -1>
  %v = and <4 x i32> %a, %w
  ret <4 x i32> %v

; CHECK-LABEL: @test17
; CHECK: xxlandc 34, 34, 35
; CHECK: blr
}

define <8 x i16> @test18(<8 x i16> %a, <8 x i16> %b) {
entry:
  %w = xor <8 x i16> %b, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  %v = and <8 x i16> %a, %w
  ret <8 x i16> %v

; CHECK-LABEL: @test18
; CHECK: xxlandc 34, 34, 35
; CHECK: blr
}

define <16 x i8> @test19(<16 x i8> %a, <16 x i8> %b) {
entry:
  %w = xor <16 x i8> %b, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %v = and <16 x i8> %a, %w
  ret <16 x i8> %v

; CHECK-LABEL: @test19
; CHECK: xxlandc 34, 34, 35
; CHECK: blr
}

define <4 x i32> @test20(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c, <4 x i32> %d) {
entry:
  %m = icmp eq <4 x i32> %c, %d
  %v = select <4 x i1> %m, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %v

; CHECK-LABEL: @test20
; CHECK: vcmpequw {{[0-9]+}}, 4, 5
; CHECK: xxsel 34, 35, 34, {{[0-9]+}}
; CHECK: blr
}

define <4 x float> @test21(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %d) {
entry:
  %m = fcmp oeq <4 x float> %c, %d
  %v = select <4 x i1> %m, <4 x float> %a, <4 x float> %b
  ret <4 x float> %v

; CHECK-LABEL: @test21
; CHECK: xvcmpeqsp [[V1:[0-9]+]], 36, 37
; CHECK: xxsel 34, 35, 34, [[V1]]
; CHECK: blr
}

define <4 x float> @test22(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %d) {
entry:
  %m = fcmp ueq <4 x float> %c, %d
  %v = select <4 x i1> %m, <4 x float> %a, <4 x float> %b
  ret <4 x float> %v

; CHECK-LABEL: @test22
; CHECK-DAG: xvcmpeqsp {{[0-9]+}}, 37, 37
; CHECK-DAG: xvcmpeqsp {{[0-9]+}}, 36, 36
; CHECK-DAG: xvcmpeqsp {{[0-9]+}}, 36, 37
; CHECK-DAG: xxlnor
; CHECK-DAG: xxlnor
; CHECK-DAG: xxlor
; CHECK-DAG: xxlor
; CHECK: xxsel 34, 35, 34, {{[0-9]+}}
; CHECK: blr
}

define <8 x i16> @test23(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c, <8 x i16> %d) {
entry:
  %m = icmp eq <8 x i16> %c, %d
  %v = select <8 x i1> %m, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %v

; CHECK-LABEL: @test23
; CHECK: vcmpequh {{[0-9]+}}, 4, 5
; CHECK: xxsel 34, 35, 34, {{[0-9]+}}
; CHECK: blr
}

define <16 x i8> @test24(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c, <16 x i8> %d) {
entry:
  %m = icmp eq <16 x i8> %c, %d
  %v = select <16 x i1> %m, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %v

; CHECK-LABEL: @test24
; CHECK: vcmpequb {{[0-9]+}}, 4, 5
; CHECK: xxsel 34, 35, 34, {{[0-9]+}}
; CHECK: blr
}

define <2 x double> @test25(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %d) {
entry:
  %m = fcmp oeq <2 x double> %c, %d
  %v = select <2 x i1> %m, <2 x double> %a, <2 x double> %b
  ret <2 x double> %v

; CHECK-LABEL: @test25
; CHECK: xvcmpeqdp [[V1:[0-9]+]], 36, 37
; CHECK: xxsel 34, 35, 34, [[V1]]
; CHECK: blr
}

define <2 x i64> @test26(<2 x i64> %a, <2 x i64> %b) {
  %v = add <2 x i64> %a, %b
  ret <2 x i64> %v

; CHECK-LABEL: @test26

; Make sure we use only two stores (one for each operand).
; CHECK: stxvd2x 35,
; CHECK: stxvd2x 34,
; CHECK-NOT: stxvd2x

; FIXME: The code quality here is not good; just make sure we do something for now.
; CHECK: add
; CHECK: add
; CHECK: blr
}

define <2 x i64> @test27(<2 x i64> %a, <2 x i64> %b) {
  %v = and <2 x i64> %a, %b
  ret <2 x i64> %v

; CHECK-LABEL: @test27
; CHECK: xxland 34, 34, 35
; CHECK: blr
}

define <2 x double> @test28(<2 x double>* %a) {
  %v = load <2 x double>* %a, align 16
  ret <2 x double> %v

; CHECK-LABEL: @test28
; CHECK: lxvd2x 34, 0, 3
; CHECK: blr
}

define void @test29(<2 x double>* %a, <2 x double> %b) {
  store <2 x double> %b, <2 x double>* %a, align 16
  ret void

; CHECK-LABEL: @test29
; CHECK: stxvd2x 34, 0, 3
; CHECK: blr
}

define <2 x double> @test28u(<2 x double>* %a) {
  %v = load <2 x double>* %a, align 8
  ret <2 x double> %v

; CHECK-LABEL: @test28u
; CHECK: lxvd2x 34, 0, 3
; CHECK: blr
}

define void @test29u(<2 x double>* %a, <2 x double> %b) {
  store <2 x double> %b, <2 x double>* %a, align 8
  ret void

; CHECK-LABEL: @test29u
; CHECK: stxvd2x 34, 0, 3
; CHECK: blr
}

define <2 x i64> @test30(<2 x i64>* %a) {
  %v = load <2 x i64>* %a, align 16
  ret <2 x i64> %v

; CHECK-LABEL: @test30
; CHECK: lxvd2x 34, 0, 3
; CHECK: blr
}

define void @test31(<2 x i64>* %a, <2 x i64> %b) {
  store <2 x i64> %b, <2 x i64>* %a, align 16
  ret void

; CHECK-LABEL: @test31
; CHECK: stxvd2x 34, 0, 3
; CHECK: blr
}

define <2 x double> @test40(<2 x i64> %a) {
  %v = uitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %v

; CHECK-LABEL: @test40
; CHECK: xvcvuxddp 34, 34
; CHECK: blr
}

define <2 x double> @test41(<2 x i64> %a) {
  %v = sitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %v

; CHECK-LABEL: @test41
; CHECK: xvcvsxddp 34, 34
; CHECK: blr
}

define <2 x i64> @test42(<2 x double> %a) {
  %v = fptoui <2 x double> %a to <2 x i64>
  ret <2 x i64> %v

; CHECK-LABEL: @test42
; CHECK: xvcvdpuxds 34, 34
; CHECK: blr
}

define <2 x i64> @test43(<2 x double> %a) {
  %v = fptosi <2 x double> %a to <2 x i64>
  ret <2 x i64> %v

; CHECK-LABEL: @test43
; CHECK: xvcvdpsxds 34, 34
; CHECK: blr
}

define <2 x float> @test44(<2 x i64> %a) {
  %v = uitofp <2 x i64> %a to <2 x float>
  ret <2 x float> %v

; CHECK-LABEL: @test44
; FIXME: The code quality here looks pretty bad.
; CHECK: blr
}

define <2 x float> @test45(<2 x i64> %a) {
  %v = sitofp <2 x i64> %a to <2 x float>
  ret <2 x float> %v

; CHECK-LABEL: @test45
; FIXME: The code quality here looks pretty bad.
; CHECK: blr
}

define <2 x i64> @test46(<2 x float> %a) {
  %v = fptoui <2 x float> %a to <2 x i64>
  ret <2 x i64> %v

; CHECK-LABEL: @test46
; FIXME: The code quality here looks pretty bad.
; CHECK: blr
}

define <2 x i64> @test47(<2 x float> %a) {
  %v = fptosi <2 x float> %a to <2 x i64>
  ret <2 x i64> %v

; CHECK-LABEL: @test47
; FIXME: The code quality here looks pretty bad.
; CHECK: blr
}

define <2 x double> @test50(double* %a) {
  %v = load double* %a, align 8
  %w = insertelement <2 x double> undef, double %v, i32 0
  %x = insertelement <2 x double> %w, double %v, i32 1
  ret <2 x double> %x

; CHECK-LABEL: @test50
; CHECK: lxvdsx 34, 0, 3
; CHECK: blr
}

define <2 x double> @test51(<2 x double> %a, <2 x double> %b) {
  %v = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 0>
  ret <2 x double> %v

; CHECK-LABEL: @test51
; CHECK: xxpermdi 34, 34, 34, 0
; CHECK: blr
}

define <2 x double> @test52(<2 x double> %a, <2 x double> %b) {
  %v = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x double> %v

; CHECK-LABEL: @test52
; CHECK: xxpermdi 34, 34, 35, 0
; CHECK: blr
}

define <2 x double> @test53(<2 x double> %a, <2 x double> %b) {
  %v = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 2, i32 0>
  ret <2 x double> %v

; CHECK-LABEL: @test53
; CHECK: xxpermdi 34, 35, 34, 0
; CHECK: blr
}

define <2 x double> @test54(<2 x double> %a, <2 x double> %b) {
  %v = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x double> %v

; CHECK-LABEL: @test54
; CHECK: xxpermdi 34, 34, 35, 2
; CHECK: blr
}

define <2 x double> @test55(<2 x double> %a, <2 x double> %b) {
  %v = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x double> %v

; CHECK-LABEL: @test55
; CHECK: xxpermdi 34, 34, 35, 3
; CHECK: blr
}

define <2 x i64> @test56(<2 x i64> %a, <2 x i64> %b) {
  %v = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i64> %v

; CHECK-LABEL: @test56
; CHECK: xxpermdi 34, 34, 35, 3
; CHECK: blr
}

define <2 x i64> @test60(<2 x i64> %a, <2 x i64> %b) {
  %v = shl <2 x i64> %a, %b
  ret <2 x i64> %v

; CHECK-LABEL: @test60
; This should scalarize, and the current code quality is not good.
; CHECK: stxvd2x
; CHECK: stxvd2x
; CHECK: sld
; CHECK: sld
; CHECK: lxvd2x
; CHECK: blr
}

define <2 x i64> @test61(<2 x i64> %a, <2 x i64> %b) {
  %v = lshr <2 x i64> %a, %b
  ret <2 x i64> %v

; CHECK-LABEL: @test61
; This should scalarize, and the current code quality is not good.
; CHECK: stxvd2x
; CHECK: stxvd2x
; CHECK: srd
; CHECK: srd
; CHECK: lxvd2x
; CHECK: blr
}

define <2 x i64> @test62(<2 x i64> %a, <2 x i64> %b) {
  %v = ashr <2 x i64> %a, %b
  ret <2 x i64> %v

; CHECK-LABEL: @test62
; This should scalarize, and the current code quality is not good.
; CHECK: stxvd2x
; CHECK: stxvd2x
; CHECK: srad
; CHECK: srad
; CHECK: lxvd2x
; CHECK: blr
}

define double @test63(<2 x double> %a) {
  %v = extractelement <2 x double> %a, i32 0
  ret double %v

; CHECK-LABEL: @test63
; CHECK: xxlor 1, 34, 34
; CHECK: blr
}

define double @test64(<2 x double> %a) {
  %v = extractelement <2 x double> %a, i32 1
  ret double %v

; CHECK-LABEL: @test64
; CHECK: xxpermdi 1, 34, 34, 2
; CHECK: blr
}

define <2 x i1> @test65(<2 x i64> %a, <2 x i64> %b) {
  %w = icmp eq <2 x i64> %a, %b
  ret <2 x i1> %w

; CHECK-LABEL: @test65
; CHECK: vcmpequw 2, 2, 3
; CHECK: blr
}

define <2 x i1> @test66(<2 x i64> %a, <2 x i64> %b) {
  %w = icmp ne <2 x i64> %a, %b
  ret <2 x i1> %w

; CHECK-LABEL: @test66
; CHECK: vcmpequw {{[0-9]+}}, 2, 3
; CHECK: xxlnor 34, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define <2 x i1> @test67(<2 x i64> %a, <2 x i64> %b) {
  %w = icmp ult <2 x i64> %a, %b
  ret <2 x i1> %w

; CHECK-LABEL: @test67
; This should scalarize, and the current code quality is not good.
; CHECK: stxvd2x
; CHECK: stxvd2x
; CHECK: cmpld
; CHECK: cmpld
; CHECK: lxvd2x
; CHECK: blr
}

define <2 x double> @test68(<2 x i32> %a) {
  %w = sitofp <2 x i32> %a to <2 x double>
  ret <2 x double> %w

; CHECK-LABEL: @test68
; CHECK: xxsldwi [[V1:[0-9]+]], 34, 34, 1
; CHECK: xvcvsxwdp 34, [[V1]]
; CHECK: blr
}

define <2 x double> @test69(<2 x i16> %a) {
  %w = sitofp <2 x i16> %a to <2 x double>
  ret <2 x double> %w

; CHECK-LABEL: @test69
; CHECK: vspltisw [[V1:[0-9]+]], 8
; CHECK: vadduwm [[V2:[0-9]+]], [[V1]], [[V1]]
; CHECK: vslw [[V3:[0-9]+]], 2, [[V2]]
; CHECK: vsraw {{[0-9]+}}, [[V3]], [[V2]]
; CHECK: xxsldwi [[V4:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}, 1
; CHECK: xvcvsxwdp 34, [[V4]]
; CHECK: blr
}

define <2 x double> @test70(<2 x i8> %a) {
  %w = sitofp <2 x i8> %a to <2 x double>
  ret <2 x double> %w

; CHECK-LABEL: @test70
; CHECK: vspltisw [[V1:[0-9]+]], 12
; CHECK: vadduwm [[V2:[0-9]+]], [[V1]], [[V1]]
; CHECK: vslw [[V3:[0-9]+]], 2, [[V2]]
; CHECK: vsraw {{[0-9]+}}, [[V3]], [[V2]]
; CHECK: xxsldwi [[V4:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}, 1
; CHECK: xvcvsxwdp 34, [[V4]]
; CHECK: blr
}

define <2 x i32> @test80(i32 %v) {
  %b1 = insertelement <2 x i32> undef, i32 %v, i32 0
  %b2 = shufflevector <2 x i32> %b1, <2 x i32> undef, <2 x i32> zeroinitializer
  %i = add <2 x i32> %b2, <i32 2, i32 3>
  ret <2 x i32> %i

; CHECK-LABEL: @test80
; CHECK-DAG: addi [[R1:[0-9]+]], 3, 3
; CHECK-DAG: addi [[R2:[0-9]+]], 1, -16
; CHECK-DAG: addi [[R3:[0-9]+]], 3, 2
; CHECK: std [[R1]], 8([[R2]])
; CHECK: std [[R3]], -16(1)
; CHECK: lxvd2x 34, 0, [[R2]]
; CHECK-NOT: stxvd2x
; CHECK: blr
}

