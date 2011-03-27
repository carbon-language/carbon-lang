// RUN: %clang_cc1 -faltivec -triple powerpc-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// Check initialization

vector int test0 = (vector int)(1);       // CHECK: @test0 = global <4 x i32> <i32 1, i32 1, i32 1, i32 1>
vector float test1 = (vector float)(1.0); // CHECK: @test1 = global <4 x float> <float 1.000000e+{{0+}}, float 1.000000e+{{0+}}, float 1.000000e+{{0+}}, float 1.000000e+{{0+}}>

void test2()
{
  vector int vi;
  vector float vf;
  vi = (vector int)(1);             // CHECK: <i32 1, i32 1, i32 1, i32 1>
  vf = (vector float)(1.0);         // CHECK: <float 1.000000e+{{0+}}, float 1.000000e+{{0+}}, float 1.000000e+{{0+}}, float 1.000000e+{{0+}}>
  vi = (vector int)(1, 2, 3, 4);    // CHECK: <i32 1, i32 2, i32 3, i32 4>
  vi = (vector int)(1, 2, 3, 4, 5); // CHECK: <i32 1, i32 2, i32 3, i32 4>

  vi = (vector int){1};             // CHECK: <i32 1, i32 0, i32 0, i32 0>
  vi = (vector int){1, 2};          // CHECK: <i32 1, i32 2, i32 0, i32 0>
  vi = (vector int){1, 2, 3, 4};    // CHECK: <i32 1, i32 2, i32 3, i32 4>

}

// Check pre/post increment/decrement
void test3() {
  vector int vi;
  vi++;                                    // CHECK: add nsw <4 x i32> {{.*}} <i32 1, i32 1, i32 1, i32 1>
  vector unsigned int vui;
  --vui;                                   // CHECK: add <4 x i32> {{.*}} <i32 -1, i32 -1, i32 -1, i32 -1>
  vector float vf;
  vf++;                                    // CHECK: fadd <4 x float> {{.*}} <float 1.000000e+{{0+}}, float 1.000000e+{{0+}}, float 1.000000e+{{0+}}, float 1.000000e+{{0+}}>
}
