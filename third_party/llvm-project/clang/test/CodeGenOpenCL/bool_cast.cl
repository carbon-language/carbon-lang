// RUN: %clang_cc1 -no-opaque-pointers %s -triple x86_64-unknown-linux-gnu -emit-llvm -o - -O0 | FileCheck %s

typedef unsigned char uchar4 __attribute((ext_vector_type(4)));
typedef unsigned int int4 __attribute((ext_vector_type(4)));
typedef float float4 __attribute((ext_vector_type(4)));

// CHECK-LABEL: define{{.*}} spir_kernel void @ker()
void kernel ker() {
  bool t = true;
  int4 vec4 = (int4)t;
// CHECK: {{%.*}} = load i8, i8* %t, align 1
// CHECK: {{%.*}} = trunc i8 {{%.*}} to i1
// CHECK: {{%.*}} = sext i1 {{%.*}} to i32
// CHECK: {{%.*}} = insertelement <4 x i32> poison, i32 {{%.*}}, i32 0
// CHECK: {{%.*}} = shufflevector <4 x i32> {{%.*}}, <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK: store <4 x i32> {{%.*}}, <4 x i32>* %vec4, align 16
  int i = (int)t;
// CHECK: {{%.*}} = load i8, i8* %t, align 1
// CHECK: {{%.*}} = trunc i8 {{%.*}} to i1
// CHECK: {{%.*}} = zext i1 {{%.*}} to i32
// CHECK: store i32 {{%.*}}, i32* %i, align 4

  uchar4 vc;
  vc = (uchar4)true;
// CHECK: store <4 x i8> <i8 -1, i8 -1, i8 -1, i8 -1>, <4 x i8>* %vc, align 4
  unsigned char c;
  c = (unsigned char)true;
// CHECK: store i8 1, i8* %c, align 1

  float4 vf;
  vf = (float4)true;
// CHECK: store <4 x float> <float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00>
}
