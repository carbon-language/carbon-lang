// REQUIRES: r600-registered-target
// RUN: %clang_cc1 -triple r600-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// CHECK-LABEL: @test_div_scale_f64
// CHECK: call { double, i1 } @llvm.AMDGPU.div.scale.f64(double %a, double %b, i1 true)
// CHECK-DAG: [[FLAG:%.+]] = extractvalue { double, i1 } %{{.+}}, 1
// CHECK-DAG: [[VAL:%.+]] = extractvalue { double, i1 } %{{.+}}, 0
// CHECK: [[FLAGEXT:%.+]] = zext i1 [[FLAG]] to i32
// CHECK: store i32 [[FLAGEXT]]
void test_div_scale_f64(global double* out, global int* flagout, double a, double b)
{
  bool flag;
  *out = __builtin_amdgpu_div_scale(a, b, true, &flag);
  *flagout = flag;
}

// CHECK-LABEL: @test_div_scale_f32
// CHECK: call { float, i1 } @llvm.AMDGPU.div.scale.f32(float %a, float %b, i1 true)
// CHECK-DAG: [[FLAG:%.+]] = extractvalue { float, i1 } %{{.+}}, 1
// CHECK-DAG: [[VAL:%.+]] = extractvalue { float, i1 } %{{.+}}, 0
// CHECK: [[FLAGEXT:%.+]] = zext i1 [[FLAG]] to i32
// CHECK: store i32 [[FLAGEXT]]
void test_div_scale_f32(global float* out, global int* flagout, float a, float b)
{
  bool flag;
  *out = __builtin_amdgpu_div_scalef(a, b, true, &flag);
  *flagout = flag;
}
