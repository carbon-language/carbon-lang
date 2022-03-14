; RUN: llc -mtriple=thumbv7-linux-gnueabihf %s -o - | FileCheck %s

define float @test_load_s32_float(i32* %addr) {
; CHECK-LABEL: test_load_s32_float:
; CHECK: ldr [[TMP:r[0-9]+]], [r0]
; CHECK: vmov [[RES_INT:s[0-9]+]], [[TMP]]
; CHECK: vcvt.f32.s32 s0, [[RES_INT]]

  %val = load i32, i32* %addr, align 1
  %res = sitofp i32 %val to float
  ret float %res
}

define double @test_load_s32_double(i32* %addr) {
; CHECK-LABEL: test_load_s32_double:
; CHECK: ldr [[TMP:r[0-9]+]], [r0]
; CHECK: vmov [[RES_INT:s[0-9]+]], [[TMP]]
; CHECK: vcvt.f64.s32 d0, [[RES_INT]]

  %val = load i32, i32* %addr, align 1
  %res = sitofp i32 %val to double
  ret double %res
}

define float @test_load_u32_float(i32* %addr) {
; CHECK-LABEL: test_load_u32_float:
; CHECK: ldr [[TMP:r[0-9]+]], [r0]
; CHECK: vmov [[RES_INT:s[0-9]+]], [[TMP]]
; CHECK: vcvt.f32.u32 s0, [[RES_INT]]

  %val = load i32, i32* %addr, align 1
  %res = uitofp i32 %val to float
  ret float %res
}

define double @test_load_u32_double(i32* %addr) {
; CHECK-LABEL: test_load_u32_double:
; CHECK: ldr [[TMP:r[0-9]+]], [r0]
; CHECK: vmov [[RES_INT:s[0-9]+]], [[TMP]]
; CHECK: vcvt.f64.u32 d0, [[RES_INT]]

  %val = load i32, i32* %addr, align 1
  %res = uitofp i32 %val to double
  ret double %res
}

define void @test_store_f32(float %in, float* %addr) {
; CHECK-LABEL: test_store_f32:
; CHECK: vmov [[TMP:r[0-9]+]], s0
; CHECK: str [[TMP]], [r0]

  store float %in, float* %addr, align 1
  ret void
}

define void @test_store_float_s32(float %in, i32* %addr) {
; CHECK-LABEL: test_store_float_s32:
; CHECK: vcvt.s32.f32 [[TMP:s[0-9]+]], s0
; CHECK: vmov [[TMP_INT:r[0-9]+]], [[TMP]]
; CHECK: str [[TMP_INT]], [r0]

  %val = fptosi float %in to i32
  store i32 %val, i32* %addr, align 1
  ret void
}

define void @test_store_double_s32(double %in, i32* %addr) {
; CHECK-LABEL: test_store_double_s32:
; CHECK: vcvt.s32.f64 [[TMP:s[0-9]+]], d0
; CHECK: vmov [[TMP_INT:r[0-9]+]], [[TMP]]
; CHECK: str [[TMP_INT]], [r0]

  %val = fptosi double %in to i32
  store i32 %val, i32* %addr, align 1
  ret void
}

define void @test_store_float_u32(float %in, i32* %addr) {
; CHECK-LABEL: test_store_float_u32:
; CHECK: vcvt.u32.f32 [[TMP:s[0-9]+]], s0
; CHECK: vmov [[TMP_INT:r[0-9]+]], [[TMP]]
; CHECK: str [[TMP_INT]], [r0]

  %val = fptoui float %in to i32
  store i32 %val, i32* %addr, align 1
  ret void
}

define void @test_store_double_u32(double %in, i32* %addr) {
; CHECK-LABEL: test_store_double_u32:
; CHECK: vcvt.u32.f64 [[TMP:s[0-9]+]], d0
; CHECK: vmov [[TMP_INT:r[0-9]+]], [[TMP]]
; CHECK: str [[TMP_INT]], [r0]

  %val = fptoui double %in to i32
  store i32 %val, i32* %addr, align 1
  ret void
}
