; RUN: llc %s -o - -verify-machineinstrs -mtriple=arm64-none-linux-gnu | FileCheck %s

; This is the analogue of AArch64's file of the same name. It's mostly testing
; some form of correct lowering occurs, the tests are a little artificial but I
; strongly suspect there's room for improved CodeGen (FIXME).

define i64 @test_sext_extr_cmp_0(<1 x i64> %v1, <1 x i64> %v2) {
; CHECK-LABEL: test_sext_extr_cmp_0:
; CHECK: cmp {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: csinc
  %1 = icmp sge <1 x i64> %v1, %v2
  %2 = extractelement <1 x i1> %1, i32 0
  %vget_lane = sext i1 %2 to i64
  ret i64 %vget_lane
}

define i64 @test_sext_extr_cmp_1(<1 x double> %v1, <1 x double> %v2) {
; CHECK-LABEL: test_sext_extr_cmp_1:
; CHECK: fcmp {{d[0-9]+}}, {{d[0-9]+}}
  %1 = fcmp oeq <1 x double> %v1, %v2
  %2 = extractelement <1 x i1> %1, i32 0
  %vget_lane = sext i1 %2 to i64
  ret i64 %vget_lane
}

define <1 x i64> @test_select_v1i1_0(<1 x i64> %v1, <1 x i64> %v2, <1 x i64> %v3) {
; CHECK-LABEL: test_select_v1i1_0:
; CHECK: cmeq d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
; CHECK: bic v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %1 = icmp eq <1 x i64> %v1, %v2
  %res = select <1 x i1> %1, <1 x i64> zeroinitializer, <1 x i64> %v3
  ret <1 x i64> %res
}

define <1 x i64> @test_select_v1i1_1(<1 x double> %v1, <1 x double> %v2, <1 x i64> %v3) {
; CHECK-LABEL: test_select_v1i1_1:
; CHECK: fcmeq d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
; CHECK: bic v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %1 = fcmp oeq <1 x double> %v1, %v2
  %res = select <1 x i1> %1, <1 x i64> zeroinitializer, <1 x i64> %v3
  ret <1 x i64> %res
}

define <1 x double> @test_select_v1i1_2(<1 x i64> %v1, <1 x i64> %v2, <1 x double> %v3) {
; CHECK-LABEL: test_select_v1i1_2:
; CHECK: cmeq d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
; CHECK: bic v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %1 = icmp eq <1 x i64> %v1, %v2
  %res = select <1 x i1> %1, <1 x double> zeroinitializer, <1 x double> %v3
  ret <1 x double> %res
}

define i32 @test_br_extr_cmp(<1 x i64> %v1, <1 x i64> %v2) {
; CHECK-LABEL: test_br_extr_cmp:
; CHECK: cmp x{{[0-9]+}}, x{{[0-9]+}}
  %1 = icmp eq <1 x i64> %v1, %v2
  %2 = extractelement <1 x i1> %1, i32 0
  br i1 %2, label %if.end, label %if.then

if.then:
  ret i32 0;

if.end:
  ret i32 1;
}
