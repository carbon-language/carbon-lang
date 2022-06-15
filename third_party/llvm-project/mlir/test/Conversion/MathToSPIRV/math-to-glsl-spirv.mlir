// RUN: mlir-opt -split-input-file -convert-math-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], []>, #spv.resource_limits<>>
} {

// CHECK-LABEL: @float32_unary_scalar
func.func @float32_unary_scalar(%arg0: f32) {
  // CHECK: spv.GLSL.Cos %{{.*}}: f32
  %0 = math.cos %arg0 : f32
  // CHECK: spv.GLSL.Exp %{{.*}}: f32
  %1 = math.exp %arg0 : f32
  // CHECK: %[[EXP:.+]] = spv.GLSL.Exp %arg0
  // CHECK: %[[ONE:.+]] = spv.Constant 1.000000e+00 : f32
  // CHECK: spv.FSub %[[EXP]], %[[ONE]]
  %2 = math.expm1 %arg0 : f32
  // CHECK: spv.GLSL.Log %{{.*}}: f32
  %3 = math.log %arg0 : f32
  // CHECK: %[[ONE:.+]] = spv.Constant 1.000000e+00 : f32
  // CHECK: %[[ADDONE:.+]] = spv.FAdd %[[ONE]], %{{.+}}
  // CHECK: spv.GLSL.Log %[[ADDONE]]
  %4 = math.log1p %arg0 : f32
  // CHECK: spv.GLSL.InverseSqrt %{{.*}}: f32
  %5 = math.rsqrt %arg0 : f32
  // CHECK: spv.GLSL.Sqrt %{{.*}}: f32
  %6 = math.sqrt %arg0 : f32
  // CHECK: spv.GLSL.Tanh %{{.*}}: f32
  %7 = math.tanh %arg0 : f32
  // CHECK: spv.GLSL.Sin %{{.*}}: f32
  %8 = math.sin %arg0 : f32
  // CHECK: spv.GLSL.FAbs %{{.*}}: f32
  %9 = math.abs %arg0 : f32
  // CHECK: spv.GLSL.Ceil %{{.*}}: f32
  %10 = math.ceil %arg0 : f32
  // CHECK: spv.GLSL.Floor %{{.*}}: f32
  %11 = math.floor %arg0 : f32
  return
}

// CHECK-LABEL: @float32_unary_vector
func.func @float32_unary_vector(%arg0: vector<3xf32>) {
  // CHECK: spv.GLSL.Cos %{{.*}}: vector<3xf32>
  %0 = math.cos %arg0 : vector<3xf32>
  // CHECK: spv.GLSL.Exp %{{.*}}: vector<3xf32>
  %1 = math.exp %arg0 : vector<3xf32>
  // CHECK: %[[EXP:.+]] = spv.GLSL.Exp %arg0
  // CHECK: %[[ONE:.+]] = spv.Constant dense<1.000000e+00> : vector<3xf32>
  // CHECK: spv.FSub %[[EXP]], %[[ONE]]
  %2 = math.expm1 %arg0 : vector<3xf32>
  // CHECK: spv.GLSL.Log %{{.*}}: vector<3xf32>
  %3 = math.log %arg0 : vector<3xf32>
  // CHECK: %[[ONE:.+]] = spv.Constant dense<1.000000e+00> : vector<3xf32>
  // CHECK: %[[ADDONE:.+]] = spv.FAdd %[[ONE]], %{{.+}}
  // CHECK: spv.GLSL.Log %[[ADDONE]]
  %4 = math.log1p %arg0 : vector<3xf32>
  // CHECK: spv.GLSL.InverseSqrt %{{.*}}: vector<3xf32>
  %5 = math.rsqrt %arg0 : vector<3xf32>
  // CHECK: spv.GLSL.Sqrt %{{.*}}: vector<3xf32>
  %6 = math.sqrt %arg0 : vector<3xf32>
  // CHECK: spv.GLSL.Tanh %{{.*}}: vector<3xf32>
  %7 = math.tanh %arg0 : vector<3xf32>
  // CHECK: spv.GLSL.Sin %{{.*}}: vector<3xf32>
  %8 = math.sin %arg0 : vector<3xf32>
  return
}

// CHECK-LABEL: @float32_binary_scalar
func.func @float32_binary_scalar(%lhs: f32, %rhs: f32) {
  // CHECK: spv.GLSL.Pow %{{.*}}: f32
  %0 = math.powf %lhs, %rhs : f32
  return
}

// CHECK-LABEL: @float32_binary_vector
func.func @float32_binary_vector(%lhs: vector<4xf32>, %rhs: vector<4xf32>) {
  // CHECK: spv.GLSL.Pow %{{.*}}: vector<4xf32>
  %0 = math.powf %lhs, %rhs : vector<4xf32>
  return
}

// CHECK-LABEL: @float32_ternary_scalar
func.func @float32_ternary_scalar(%a: f32, %b: f32, %c: f32) {
  // CHECK: spv.GLSL.Fma %{{.*}}: f32
  %0 = math.fma %a, %b, %c : f32
  return
}

// CHECK-LABEL: @float32_ternary_vector
func.func @float32_ternary_vector(%a: vector<4xf32>, %b: vector<4xf32>,
                            %c: vector<4xf32>) {
  // CHECK: spv.GLSL.Fma %{{.*}}: vector<4xf32>
  %0 = math.fma %a, %b, %c : vector<4xf32>
  return
}

// CHECK-LABEL: @ctlz_scalar
//  CHECK-SAME: (%[[VAL:.+]]: i32)
func.func @ctlz_scalar(%val: i32) -> i32 {
  // CHECK: %[[V31:.+]] = spv.Constant 31 : i32
  // CHECK: %[[MSB:.+]] = spv.GLSL.FindUMsb %[[VAL]] : i32
  // CHECK: %[[SUB:.+]] = spv.ISub %[[V31]], %[[MSB]] : i32
  // CHECK: return %[[SUB]]
  %0 = math.ctlz %val : i32
  return %0 : i32
}

// CHECK-LABEL: @ctlz_vector1
func.func @ctlz_vector1(%val: vector<1xi32>) -> vector<1xi32> {
  // CHECK: spv.GLSL.FindUMsb
  // CHECK: spv.ISub
  %0 = math.ctlz %val : vector<1xi32>
  return %0 : vector<1xi32>
}

// CHECK-LABEL: @ctlz_vector2
//  CHECK-SAME: (%[[VAL:.+]]: vector<2xi32>)
func.func @ctlz_vector2(%val: vector<2xi32>) -> vector<2xi32> {
  // CHECK-DAG: %[[V31:.+]] = spv.Constant dense<31> : vector<2xi32>
  // CHECK: %[[MSB:.+]] = spv.GLSL.FindUMsb %[[VAL]] : vector<2xi32>
  // CHECK: %[[SUB:.+]] = spv.ISub %[[V31]], %[[MSB]] : vector<2xi32>
  // CHECK: return %[[SUB]]
  %0 = math.ctlz %val : vector<2xi32>
  return %0 : vector<2xi32>
}

} // end module

// -----

module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader, Int64, Int16], []>, #spv.resource_limits<>>
} {

// CHECK-LABEL: @ctlz_scalar
func.func @ctlz_scalar(%val: i64) -> i64 {
  // CHECK: math.ctlz
  %0 = math.ctlz %val : i64
  return %0 : i64
}

// CHECK-LABEL: @ctlz_vector2
func.func @ctlz_vector2(%val: vector<2xi16>) -> vector<2xi16> {
  // CHECK: math.ctlz
  %0 = math.ctlz %val : vector<2xi16>
  return %0 : vector<2xi16>
}

} // end module
