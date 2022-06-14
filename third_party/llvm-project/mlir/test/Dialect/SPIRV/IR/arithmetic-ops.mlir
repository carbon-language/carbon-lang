// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.FAdd
//===----------------------------------------------------------------------===//

func.func @fadd_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FAdd
  %0 = spv.FAdd %arg, %arg : f32
  return %0 : f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.FDiv
//===----------------------------------------------------------------------===//

func.func @fdiv_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FDiv
  %0 = spv.FDiv %arg, %arg : f32
  return %0 : f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.FMod
//===----------------------------------------------------------------------===//

func.func @fmod_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FMod
  %0 = spv.FMod %arg, %arg : f32
  return %0 : f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.FMul
//===----------------------------------------------------------------------===//

func.func @fmul_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FMul
  %0 = spv.FMul %arg, %arg : f32
  return %0 : f32
}

func.func @fmul_vector(%arg: vector<4xf32>) -> vector<4xf32> {
  // CHECK: spv.FMul
  %0 = spv.FMul %arg, %arg : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

func.func @fmul_i32(%arg: i32) -> i32 {
  // expected-error @+1 {{operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %0 = spv.FMul %arg, %arg : i32
  return %0 : i32
}

// -----

func.func @fmul_bf16(%arg: bf16) -> bf16 {
  // expected-error @+1 {{operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %0 = spv.FMul %arg, %arg : bf16
  return %0 : bf16
}

// -----

func.func @fmul_tensor(%arg: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %0 = spv.FMul %arg, %arg : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.FNegate
//===----------------------------------------------------------------------===//

func.func @fnegate_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FNegate
  %0 = spv.FNegate %arg : f32
  return %0 : f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.FRem
//===----------------------------------------------------------------------===//

func.func @frem_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FRem
  %0 = spv.FRem %arg, %arg : f32
  return %0 : f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.FSub
//===----------------------------------------------------------------------===//

func.func @fsub_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FSub
  %0 = spv.FSub %arg, %arg : f32
  return %0 : f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.IAdd
//===----------------------------------------------------------------------===//

func.func @iadd_scalar(%arg: i32) -> i32 {
  // CHECK: spv.IAdd
  %0 = spv.IAdd %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.IMul
//===----------------------------------------------------------------------===//

func.func @imul_scalar(%arg: i32) -> i32 {
  // CHECK: spv.IMul
  %0 = spv.IMul %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.ISub
//===----------------------------------------------------------------------===//

func.func @isub_scalar(%arg: i32) -> i32 {
  // CHECK: spv.ISub
  %0 = spv.ISub %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.SDiv
//===----------------------------------------------------------------------===//

func.func @sdiv_scalar(%arg: i32) -> i32 {
  // CHECK: spv.SDiv
  %0 = spv.SDiv %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.SMod
//===----------------------------------------------------------------------===//

func.func @smod_scalar(%arg: i32) -> i32 {
  // CHECK: spv.SMod
  %0 = spv.SMod %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.SNegate
//===----------------------------------------------------------------------===//

func.func @snegate_scalar(%arg: i32) -> i32 {
  // CHECK: spv.SNegate
  %0 = spv.SNegate %arg : i32
  return %0 : i32
}

// -----
//===----------------------------------------------------------------------===//
// spv.SRem
//===----------------------------------------------------------------------===//

func.func @srem_scalar(%arg: i32) -> i32 {
  // CHECK: spv.SRem
  %0 = spv.SRem %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.UDiv
//===----------------------------------------------------------------------===//

func.func @udiv_scalar(%arg: i32) -> i32 {
  // CHECK: spv.UDiv
  %0 = spv.UDiv %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.UMod
//===----------------------------------------------------------------------===//

func.func @umod_scalar(%arg: i32) -> i32 {
  // CHECK: spv.UMod
  %0 = spv.UMod %arg, %arg : i32
  return %0 : i32
}

// -----
//===----------------------------------------------------------------------===//
// spv.VectorTimesScalar
//===----------------------------------------------------------------------===//

func.func @vector_times_scalar(%vector: vector<4xf32>, %scalar: f32) -> vector<4xf32> {
  // CHECK: spv.VectorTimesScalar %{{.+}}, %{{.+}} : (vector<4xf32>, f32) -> vector<4xf32>
  %0 = spv.VectorTimesScalar %vector, %scalar : (vector<4xf32>, f32) -> vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

func.func @vector_times_scalar(%vector: vector<4xf32>, %scalar: f16) -> vector<4xf32> {
  // expected-error @+1 {{scalar operand and result element type match}}
  %0 = spv.VectorTimesScalar %vector, %scalar : (vector<4xf32>, f16) -> vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

func.func @vector_times_scalar(%vector: vector<4xf32>, %scalar: f32) -> vector<3xf32> {
  // expected-error @+1 {{vector operand and result type mismatch}}
  %0 = spv.VectorTimesScalar %vector, %scalar : (vector<4xf32>, f32) -> vector<3xf32>
  return %0 : vector<3xf32>
}
