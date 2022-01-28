// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @foo() -> () "None" {
    // CHECK: {{%.*}} = spv.Undef : f32
    // CHECK-NEXT: {{%.*}} = spv.Undef : f32
    %0 = spv.Undef : f32
    %1 = spv.Undef : f32
    %2 = spv.FAdd %0, %1 : f32
    // CHECK: {{%.*}} = spv.Undef : vector<4xi32>
    %3 = spv.Undef : vector<4xi32>
    %4 = spv.CompositeExtract %3[1 : i32] : vector<4xi32>
    // CHECK: {{%.*}} = spv.Undef : !spv.array<4 x !spv.array<4 x i32>>
    %5 = spv.Undef : !spv.array<4x!spv.array<4xi32>>
    %6 = spv.CompositeExtract %5[1 : i32, 2 : i32] : !spv.array<4x!spv.array<4xi32>>
    // CHECK: {{%.*}} = spv.Undef : !spv.ptr<!spv.struct<(f32)>, StorageBuffer>
    %7 = spv.Undef : !spv.ptr<!spv.struct<(f32)>, StorageBuffer>
    %8 = spv.Constant 0 : i32
    %9 = spv.AccessChain %7[%8] : !spv.ptr<!spv.struct<(f32)>, StorageBuffer>, i32
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: spv.func {{@.*}}
  spv.func @ignore_unused_undef() -> () "None" {
    // CHECK-NEXT: spv.Return
    %0 = spv.Undef : f32
    spv.Return
  }
}
