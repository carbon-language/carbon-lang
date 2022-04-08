// RUN: mlir-opt %s -convert-vector-to-llvm -split-input-file | FileCheck %s


func @bitcast_f32_to_i32_vector_0d(%input: vector<f32>) -> vector<i32> {
  %0 = vector.bitcast %input : vector<f32> to vector<i32>
  return %0 : vector<i32>
}

// CHECK-LABEL: @bitcast_f32_to_i32_vector_0d
// CHECK-SAME:  %[[input:.*]]: vector<f32>
// CHECK:       %[[vec_f32_1d:.*]] = builtin.unrealized_conversion_cast %[[input]] : vector<f32> to vector<1xf32>
// CHECK:       %[[vec_i32_1d:.*]] = llvm.bitcast %[[vec_f32_1d]] : vector<1xf32> to vector<1xi32>
// CHECK:       %[[vec_i32_0d:.*]] = builtin.unrealized_conversion_cast %[[vec_i32_1d]] : vector<1xi32> to vector<i32>
// CHECK:       return %[[vec_i32_0d]] : vector<i32>

// -----

func @bitcast_f32_to_i32_vector(%input: vector<16xf32>) -> vector<16xi32> {
  %0 = vector.bitcast %input : vector<16xf32> to vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL: @bitcast_f32_to_i32_vector
// CHECK-SAME:  %[[input:.*]]: vector<16xf32>
// CHECK:       llvm.bitcast %[[input]] : vector<16xf32> to vector<16xi32>

// -----

func @bitcast_i8_to_f32_vector(%input: vector<64xi8>) -> vector<16xf32> {
  %0 = vector.bitcast %input : vector<64xi8> to vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @bitcast_i8_to_f32_vector
// CHECK-SAME:  %[[input:.*]]: vector<64xi8>
// CHECK:       llvm.bitcast %[[input]] : vector<64xi8> to vector<16xf32>

// -----

func @bitcast_index_to_i8_vector(%input: vector<16xindex>) -> vector<128xi8> {
  %0 = vector.bitcast %input : vector<16xindex> to vector<128xi8>
  return %0 : vector<128xi8>
}

// CHECK-LABEL: @bitcast_index_to_i8_vector
// CHECK-SAME:  %[[input:.*]]: vector<16xindex>
// CHECK:       %[[T0:.*]] = builtin.unrealized_conversion_cast %[[input]] : vector<16xindex> to vector<16xi64>
// CHECK:       llvm.bitcast %[[T0]] : vector<16xi64> to vector<128xi8>

// -----

func @broadcast_vec0d_from_f32(%arg0: f32) -> vector<f32> {
  %0 = vector.broadcast %arg0 : f32 to vector<f32>
  return %0 : vector<f32>
}
// CHECK-LABEL: @broadcast_vec0d_from_f32
// CHECK-SAME:  %[[A:.*]]: f32)
// CHECK:       %[[T0:.*]] = llvm.insertelement %[[A]]
// CHECK:       %[[T1:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<1xf32> to vector<f32>
// CHECK:       return %[[T1]] : vector<f32>

// -----

func @broadcast_vec0d_from_vec0d(%arg0: vector<f32>) -> vector<f32> {
  %0 = vector.broadcast %arg0 : vector<f32> to vector<f32>
  return %0 : vector<f32>
}
// CHECK-LABEL: @broadcast_vec0d_from_vec0d(
// CHECK-SAME:  %[[A:.*]]: vector<f32>)
// CHECK:       return %[[A]] : vector<f32>

// -----

func @broadcast_vec1d_from_f32(%arg0: f32) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2xf32>
  return %0 : vector<2xf32>
}
// CHECK-LABEL: @broadcast_vec1d_from_f32
// CHECK-SAME:  %[[A:.*]]: f32)
// CHECK:       %[[T0:.*]] = llvm.insertelement %[[A]]
// CHECK:       %[[T1:.*]] = llvm.shufflevector %[[T0]]
// CHECK:       return %[[T1]] : vector<2xf32>

// -----

func @broadcast_vec1d_from_index(%arg0: index) -> vector<2xindex> {
  %0 = vector.broadcast %arg0 : index to vector<2xindex>
  return %0 : vector<2xindex>
}
// CHECK-LABEL: @broadcast_vec1d_from_index
// CHECK-SAME:  %[[A:.*]]: index)
// CHECK:       %[[A1:.*]] = builtin.unrealized_conversion_cast %[[A]] : index to i64
// CHECK:       %[[T0:.*]] = llvm.insertelement %[[A1]]
// CHECK:       %[[T1:.*]] = llvm.shufflevector %[[T0]]
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<2xi64> to vector<2xindex>
// CHECK:       return %[[T2]] : vector<2xindex>

// -----

func @broadcast_vec2d_from_scalar(%arg0: f32) -> vector<2x3xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3xf32>
  return %0 : vector<2x3xf32>
}
// CHECK-LABEL: @broadcast_vec2d_from_scalar(
// CHECK-SAME:  %[[A:.*]]: f32)
// CHECK:       %[[T0:.*]] = llvm.insertelement %[[A]]
// CHECK:       %[[T1:.*]] = llvm.shufflevector %[[T0]]
// CHECK:       %[[T2:.*]] = llvm.insertvalue %[[T1]], %{{.*}}[0] : !llvm.array<2 x vector<3xf32>> 
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[T1]], %{{.*}}[1] : !llvm.array<2 x vector<3xf32>> 
// CHECK:       %[[T4:.*]] = builtin.unrealized_conversion_cast %[[T3]] : !llvm.array<2 x vector<3xf32>> to vector<2x3xf32>
// CHECK:       return %[[T4]] : vector<2x3xf32>

// -----

func @broadcast_vec3d_from_scalar(%arg0: f32) -> vector<2x3x4xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3x4xf32>
  return %0 : vector<2x3x4xf32>
}
// CHECK-LABEL: @broadcast_vec3d_from_scalar(
// CHECK-SAME:  %[[A:.*]]: f32)
// CHECK:       %[[T0:.*]] = llvm.insertelement %[[A]]
// CHECK:       %[[T1:.*]] = llvm.shufflevector %[[T0]]
// CHECK:       %[[T2:.*]] = llvm.insertvalue %[[T1]], %{{.*}}[0, 0] : !llvm.array<2 x array<3 x vector<4xf32>>>
// ...
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[T1]], %{{.*}}[1, 2] : !llvm.array<2 x array<3 x vector<4xf32>>>
// CHECK:       %[[T4:.*]] = builtin.unrealized_conversion_cast %[[T3]] : !llvm.array<2 x array<3 x vector<4xf32>>> to vector<2x3x4xf32>
// CHECK:       return %[[T4]] : vector<2x3x4xf32>

// -----

func @broadcast_vec1d_from_vec1d(%arg0: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<2xf32>
  return %0 : vector<2xf32>
}
// CHECK-LABEL: @broadcast_vec1d_from_vec1d(
// CHECK-SAME:  %[[A:.*]]: vector<2xf32>)
// CHECK:       return %[[A]] : vector<2xf32>

// -----

func @broadcast_vec2d_from_vec0d(%arg0: vector<f32>) -> vector<3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<f32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}
// CHECK-LABEL: @broadcast_vec2d_from_vec0d(
// CHECK-SAME:  %[[A:.*]]: vector<f32>)
//       CHECK: %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<f32> to vector<1xf32>
//       CHECK: %[[T1:.*]] = arith.constant dense<0.000000e+00> : vector<3x2xf32>
//       CHECK: %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<3x2xf32> to !llvm.array<3 x vector<2xf32>>
//       CHECK: %[[T4:.*]] = llvm.mlir.constant(0 : index) : i64
//       CHECK: %[[T5:.*]] = llvm.extractelement %[[T0]][%[[T4]] : i64] : vector<1xf32>
//       CHECK: %[[T6Insert:.*]] = llvm.insertelement %[[T5]]
//       CHECK: %[[T6:.*]] = llvm.shufflevector %[[T6Insert]]
//       CHECK: %[[T7:.*]] = llvm.insertvalue %[[T6]], %[[T2]][0] : !llvm.array<3 x vector<2xf32>>
//       CHECK: %[[T8:.*]] = llvm.insertvalue %[[T6]], %[[T7]][1] : !llvm.array<3 x vector<2xf32>>
//       CHECK: %[[T9:.*]] = llvm.insertvalue %[[T6]], %[[T8]][2] : !llvm.array<3 x vector<2xf32>>
//       CHECK: %[[T10:.*]] = builtin.unrealized_conversion_cast %[[T9]] : !llvm.array<3 x vector<2xf32>> to vector<3x2xf32>
//       CHECK: return %[[T10]] : vector<3x2xf32>

// -----

func @broadcast_vec2d_from_vec1d(%arg0: vector<2xf32>) -> vector<3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}
// CHECK-LABEL: @broadcast_vec2d_from_vec1d(
// CHECK-SAME:  %[[A:.*]]: vector<2xf32>)
// CHECK:       %[[T0:.*]] = arith.constant dense<0.000000e+00> : vector<3x2xf32>
// CHECK:       %[[T1:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<3x2xf32> to !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T2:.*]] = llvm.insertvalue %[[A]], %[[T1]][0] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[A]], %[[T2]][1] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T4:.*]] = llvm.insertvalue %[[A]], %[[T3]][2] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T5:.*]] = builtin.unrealized_conversion_cast %[[T4]] : !llvm.array<3 x vector<2xf32>> to vector<3x2xf32>
// CHECK:       return %[[T5]] : vector<3x2xf32>

// -----

func @broadcast_vec2d_from_index_vec1d(%arg0: vector<2xindex>) -> vector<3x2xindex> {
  %0 = vector.broadcast %arg0 : vector<2xindex> to vector<3x2xindex>
  return %0 : vector<3x2xindex>
}
// CHECK-LABEL: @broadcast_vec2d_from_index_vec1d(
// CHECK-SAME:  %[[A:.*]]: vector<2xindex>)
// CHECK:       %[[T1:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<2xindex> to vector<2xi64>
// CHECK:       %[[T0:.*]] = arith.constant dense<0> : vector<3x2xindex>
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<3x2xindex> to !llvm.array<3 x vector<2xi64>>
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[T1]], %[[T2]][0] : !llvm.array<3 x vector<2xi64>>

// CHECK:       %[[T4:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !llvm.array<3 x vector<2xi64>> to vector<3x2xindex>
// CHECK:       return %[[T4]] : vector<3x2xindex>

// -----

func @broadcast_vec3d_from_vec1d(%arg0: vector<2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}
// CHECK-LABEL: @broadcast_vec3d_from_vec1d(
// CHECK-SAME:  %[[A:.*]]: vector<2xf32>)
// CHECK:       %[[T0:.*]] = arith.constant dense<0.000000e+00> : vector<3x2xf32>
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<3x2xf32> to !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T1:.*]] = arith.constant dense<0.000000e+00> : vector<4x3x2xf32>
// CHECK:       %[[T6:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<4x3x2xf32> to !llvm.array<4 x array<3 x vector<2xf32>>>

// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[A]], %[[T2]][0] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T4:.*]] = llvm.insertvalue %[[A]], %[[T3]][1] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T5:.*]] = llvm.insertvalue %[[A]], %[[T4]][2] : !llvm.array<3 x vector<2xf32>>

// CHECK:       %[[T7:.*]] = llvm.insertvalue %[[T5]], %[[T6]][0] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T8:.*]] = llvm.insertvalue %[[T5]], %[[T7]][1] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T9:.*]] = llvm.insertvalue %[[T5]], %[[T8]][2] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T10:.*]] = llvm.insertvalue %[[T5]], %[[T9]][3] : !llvm.array<4 x array<3 x vector<2xf32>>>

// CHECK:       %[[T11:.*]] = builtin.unrealized_conversion_cast %[[T10]] : !llvm.array<4 x array<3 x vector<2xf32>>> to vector<4x3x2xf32>
// CHECK:       return %[[T11]] : vector<4x3x2xf32>

// -----

func @broadcast_vec3d_from_vec2d(%arg0: vector<3x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<3x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}
// CHECK-LABEL: @broadcast_vec3d_from_vec2d(
// CHECK-SAME:  %[[A:.*]]: vector<3x2xf32>)
// CHECK:       %[[T1:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<3x2xf32> to !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T0:.*]] = arith.constant dense<0.000000e+00> : vector<4x3x2xf32>
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<4x3x2xf32> to !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[T1]], %[[T2]][0] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T5:.*]] = llvm.insertvalue %[[T1]], %[[T3]][1] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T7:.*]] = llvm.insertvalue %[[T1]], %[[T5]][2] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T9:.*]] = llvm.insertvalue %[[T1]], %[[T7]][3] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T10:.*]] = builtin.unrealized_conversion_cast %[[T9]] : !llvm.array<4 x array<3 x vector<2xf32>>> to vector<4x3x2xf32>
// CHECK:       return %[[T10]] : vector<4x3x2xf32>


// -----

func @broadcast_stretch(%arg0: vector<1xf32>) -> vector<4xf32> {
  %0 = vector.broadcast %arg0 : vector<1xf32> to vector<4xf32>
  return %0 : vector<4xf32>
}
// CHECK-LABEL: @broadcast_stretch(
// CHECK-SAME:  %[[A:.*]]: vector<1xf32>)
// CHECK:       %[[T1:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T2:.*]] = llvm.extractelement %[[A]]{{\[}}%[[T1]] : i64] : vector<1xf32>
// CHECK:       %[[T3:.*]] = llvm.insertelement %[[T2]]
// CHECK:       %[[T4:.*]] = llvm.shufflevector %[[T3]]
// CHECK:       return %[[T4]] : vector<4xf32>

// -----

func @broadcast_stretch_at_start(%arg0: vector<1x4xf32>) -> vector<3x4xf32> {
  %0 = vector.broadcast %arg0 : vector<1x4xf32> to vector<3x4xf32>
  return %0 : vector<3x4xf32>
}
// CHECK-LABEL: @broadcast_stretch_at_start(
// CHECK-SAME:  %[[A:.*]]: vector<1x4xf32>)
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<1x4xf32> to !llvm.array<1 x vector<4xf32>>
// CHECK:       %[[T1:.*]] = arith.constant dense<0.000000e+00> : vector<3x4xf32>
// CHECK:       %[[T4:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<3x4xf32> to !llvm.array<3 x vector<4xf32>>
// CHECK:       %[[T3:.*]] = llvm.extractvalue %[[T2]][0] : !llvm.array<1 x vector<4xf32>>
// CHECK:       %[[T5:.*]] = llvm.insertvalue %[[T3]], %[[T4]][0] : !llvm.array<3 x vector<4xf32>>
// CHECK:       %[[T6:.*]] = llvm.insertvalue %[[T3]], %[[T5]][1] : !llvm.array<3 x vector<4xf32>>
// CHECK:       %[[T7:.*]] = llvm.insertvalue %[[T3]], %[[T6]][2] : !llvm.array<3 x vector<4xf32>>
// CHECK:       %[[T8:.*]] = builtin.unrealized_conversion_cast %[[T7]] : !llvm.array<3 x vector<4xf32>> to vector<3x4xf32>
// CHECK:       return %[[T8]] : vector<3x4xf32>

// -----

func @broadcast_stretch_at_end(%arg0: vector<4x1xf32>) -> vector<4x3xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1xf32> to vector<4x3xf32>
  return %0 : vector<4x3xf32>
}
// CHECK-LABEL: @broadcast_stretch_at_end(
// CHECK-SAME:  %[[A:.*]]: vector<4x1xf32>)
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<4x1xf32> to !llvm.array<4 x vector<1xf32>>
// CHECK:       %[[T1:.*]] = arith.constant dense<0.000000e+00> : vector<4x3xf32>
// CHECK:       %[[T7:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<4x3xf32> to !llvm.array<4 x vector<3xf32>>
// CHECK:       %[[T3:.*]] = llvm.extractvalue %[[T2]][0] : !llvm.array<4 x vector<1xf32>>
// CHECK:       %[[T4:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T5:.*]] = llvm.extractelement %[[T3]]{{\[}}%[[T4]] : i64] : vector<1xf32>
// CHECK:       %[[T6Insert:.*]] = llvm.insertelement %[[T5]]
// CHECK:       %[[T6:.*]] = llvm.shufflevector %[[T6Insert]]
// CHECK:       %[[T8:.*]] = llvm.insertvalue %[[T6]], %[[T7]][0] : !llvm.array<4 x vector<3xf32>>
// CHECK:       %[[T10:.*]] = llvm.extractvalue %[[T2]][1] : !llvm.array<4 x vector<1xf32>>
// CHECK:       %[[T11:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T12:.*]] = llvm.extractelement %[[T10]]{{\[}}%[[T11]] : i64] : vector<1xf32>
// CHECK:       %[[T13Insert:.*]] = llvm.insertelement %[[T12]]
// CHECK:       %[[T13:.*]] = llvm.shufflevector %[[T13Insert]]
// CHECK:       %[[T14:.*]] = llvm.insertvalue %[[T13]], %[[T8]][1] : !llvm.array<4 x vector<3xf32>>
// CHECK:       %[[T16:.*]] = llvm.extractvalue %[[T2]][2] : !llvm.array<4 x vector<1xf32>>
// CHECK:       %[[T17:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T18:.*]] = llvm.extractelement %[[T16]]{{\[}}%[[T17]] : i64] : vector<1xf32>
// CHECK:       %[[T19Insert:.*]] = llvm.insertelement %[[T18]]
// CHECK:       %[[T19:.*]] = llvm.shufflevector %[[T19Insert]]
// CHECK:       %[[T20:.*]] = llvm.insertvalue %[[T19]], %[[T14]][2] : !llvm.array<4 x vector<3xf32>>
// CHECK:       %[[T22:.*]] = llvm.extractvalue %[[T2]][3] : !llvm.array<4 x vector<1xf32>>
// CHECK:       %[[T23:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T24:.*]] = llvm.extractelement %[[T22]]{{\[}}%[[T23]] : i64] : vector<1xf32>
// CHECK:       %[[T25Insert:.*]] = llvm.insertelement %[[T24]]
// CHECK:       %[[T25:.*]] = llvm.shufflevector %[[T25Insert]]
// CHECK:       %[[T26:.*]] = llvm.insertvalue %[[T25]], %[[T20]][3] : !llvm.array<4 x vector<3xf32>>
// CHECK:       %[[T27:.*]] = builtin.unrealized_conversion_cast %[[T26]] : !llvm.array<4 x vector<3xf32>> to vector<4x3xf32>
// CHECK:       return %[[T27]] : vector<4x3xf32>

// -----

func @broadcast_stretch_in_middle(%arg0: vector<4x1x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}
// CHECK-LABEL: @broadcast_stretch_in_middle(
// CHECK-SAME:  %[[A:.*]]: vector<4x1x2xf32>) -> vector<4x3x2xf32> {
// CHECK:       %[[T3:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<4x1x2xf32> to !llvm.array<4 x array<1 x vector<2xf32>>>
// CHECK:       %[[T1:.*]] = arith.constant dense<0.000000e+00> : vector<4x3x2xf32>
// CHECK:       %[[T9:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<4x3x2xf32> to !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T2:.*]] = arith.constant dense<0.000000e+00> : vector<3x2xf32>
// CHECK:       %[[T5:.*]] = builtin.unrealized_conversion_cast %[[T2]] : vector<3x2xf32> to !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T4:.*]] = llvm.extractvalue %[[T3]][0, 0] : !llvm.array<4 x array<1 x vector<2xf32>>>
// CHECK:       %[[T6:.*]] = llvm.insertvalue %[[T4]], %[[T5]][0] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T7:.*]] = llvm.insertvalue %[[T4]], %[[T6]][1] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T8:.*]] = llvm.insertvalue %[[T4]], %[[T7]][2] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T10:.*]] = llvm.insertvalue %[[T8]], %[[T9]][0] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T12:.*]] = llvm.extractvalue %[[T3]][1, 0] : !llvm.array<4 x array<1 x vector<2xf32>>>
// CHECK:       %[[T14:.*]] = llvm.insertvalue %[[T12]], %[[T5]][0] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T15:.*]] = llvm.insertvalue %[[T12]], %[[T14]][1] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T16:.*]] = llvm.insertvalue %[[T12]], %[[T15]][2] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T17:.*]] = llvm.insertvalue %[[T16]], %[[T10]][1] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T19:.*]] = llvm.extractvalue %[[T3]][2, 0] : !llvm.array<4 x array<1 x vector<2xf32>>>
// CHECK:       %[[T21:.*]] = llvm.insertvalue %[[T19]], %[[T5]][0] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T22:.*]] = llvm.insertvalue %[[T19]], %[[T21]][1] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T23:.*]] = llvm.insertvalue %[[T19]], %[[T22]][2] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T24:.*]] = llvm.insertvalue %[[T23]], %[[T17]][2] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T26:.*]] = llvm.extractvalue %[[T3]][3, 0] : !llvm.array<4 x array<1 x vector<2xf32>>>
// CHECK:       %[[T28:.*]] = llvm.insertvalue %[[T26]], %[[T5]][0] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T29:.*]] = llvm.insertvalue %[[T26]], %[[T28]][1] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T30:.*]] = llvm.insertvalue %[[T26]], %[[T29]][2] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T31:.*]] = llvm.insertvalue %[[T30]], %[[T24]][3] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T32:.*]] = builtin.unrealized_conversion_cast %[[T31]] : !llvm.array<4 x array<3 x vector<2xf32>>> to vector<4x3x2xf32>
// CHECK:       return %[[T32]] : vector<4x3x2xf32>

// -----

func @outerproduct(%arg0: vector<2xf32>, %arg1: vector<3xf32>) -> vector<2x3xf32> {
  %2 = vector.outerproduct %arg0, %arg1 : vector<2xf32>, vector<3xf32>
  return %2 : vector<2x3xf32>
}
// CHECK-LABEL: @outerproduct(
// CHECK-SAME:  %[[A:.*]]: vector<2xf32>,
// CHECK-SAME:  %[[B:.*]]: vector<3xf32>)
// CHECK:       %[[T2:.*]] = arith.constant dense<0.000000e+00> : vector<2x3xf32>
// CHECK:       %[[T7:.*]] = builtin.unrealized_conversion_cast %[[T2]] : vector<2x3xf32> to !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T4:.*]] = llvm.extractelement %[[A]]{{\[}}%[[T3]] : i64] : vector<2xf32>
// CHECK:       %[[T5Insert:.*]] = llvm.insertelement %[[T4]]
// CHECK:       %[[T5:.*]] = llvm.shufflevector %[[T5Insert]]
// CHECK:       %[[T6:.*]] = arith.mulf %[[T5]], %[[B]] : vector<3xf32>
// CHECK:       %[[T8:.*]] = llvm.insertvalue %[[T6]], %[[T7]][0] : !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T9:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:       %[[T10:.*]] = llvm.extractelement %[[A]]{{\[}}%[[T9]] : i64] : vector<2xf32>
// CHECK:       %[[T11Insert:.*]] = llvm.insertelement %[[T10]]
// CHECK:       %[[T11:.*]] = llvm.shufflevector %[[T11Insert]]
// CHECK:       %[[T12:.*]] = arith.mulf %[[T11]], %[[B]] : vector<3xf32>
// CHECK:       %[[T13:.*]] = llvm.insertvalue %[[T12]], %[[T8]][1] : !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T14:.*]] = builtin.unrealized_conversion_cast %[[T13]] : !llvm.array<2 x vector<3xf32>> to vector<2x3xf32>
// CHECK:       return %[[T14]] : vector<2x3xf32>

// -----

func @outerproduct_index(%arg0: vector<2xindex>, %arg1: vector<3xindex>) -> vector<2x3xindex> {
  %2 = vector.outerproduct %arg0, %arg1 : vector<2xindex>, vector<3xindex>
  return %2 : vector<2x3xindex>
}
// CHECK-LABEL: @outerproduct_index(
// CHECK-SAME:  %[[A:.*]]: vector<2xindex>,
// CHECK-SAME:  %[[B:.*]]: vector<3xindex>)
// CHECK:       %[[T1:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<2xindex> to vector<2xi64>
// CHECK:       %[[T0:.*]] = arith.constant dense<0> : vector<2x3xindex>
// CHECK:       %[[T8:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<2x3xindex> to !llvm.array<2 x vector<3xi64>>
// CHECK:       %[[T2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T3:.*]] = llvm.extractelement %[[T1]]{{\[}}%[[T2]] : i64] : vector<2xi64>
// CHECK:       %[[T4:.*]] = llvm.insertelement %[[T3]]
// CHECK:       %[[T5:.*]] = llvm.shufflevector %[[T4]]
// CHECK:       %[[T5Cast:.*]] = builtin.unrealized_conversion_cast %[[T5]] : vector<3xi64> to vector<3xindex>
// CHECK:       %[[T6:.*]] = arith.muli %[[T5Cast]], %[[B]] : vector<3xindex>
// CHECK:       %[[T7:.*]] = builtin.unrealized_conversion_cast %[[T6]] : vector<3xindex> to vector<3xi64>
// CHECK:       %{{.*}} = llvm.insertvalue %[[T7]], %[[T8]][0] : !llvm.array<2 x vector<3xi64>>

// -----

func @outerproduct_add(%arg0: vector<2xf32>, %arg1: vector<3xf32>, %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %2 = vector.outerproduct %arg0, %arg1, %arg2 : vector<2xf32>, vector<3xf32>
  return %2 : vector<2x3xf32>
}
// CHECK-LABEL: @outerproduct_add(
// CHECK-SAME:  %[[A:.*]]: vector<2xf32>,
// CHECK-SAME:  %[[B:.*]]: vector<3xf32>,
// CHECK-SAME:  %[[C:.*]]: vector<2x3xf32>) -> vector<2x3xf32>
// CHECK:       %[[T7:.*]] = builtin.unrealized_conversion_cast %[[C]] : vector<2x3xf32> to !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T3:.*]] = arith.constant dense<0.000000e+00> : vector<2x3xf32>
// CHECK:       %[[T10:.*]] = builtin.unrealized_conversion_cast %[[T3]] : vector<2x3xf32> to !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T4:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T5:.*]] = llvm.extractelement %[[A]]{{\[}}%[[T4]] : i64] : vector<2xf32>
// CHECK:       %[[T6Insert:.*]] = llvm.insertelement %[[T5]]
// CHECK:       %[[T6:.*]] = llvm.shufflevector %[[T6Insert]]
// CHECK:       %[[T8:.*]] = llvm.extractvalue %[[T7]][0] : !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T9:.*]] = "llvm.intr.fmuladd"(%[[T6]], %[[B]], %[[T8]]) : (vector<3xf32>, vector<3xf32>, vector<3xf32>) -> vector<3xf32>
// CHECK:       %[[T11:.*]] = llvm.insertvalue %[[T9]], %[[T10]][0] : !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T12:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:       %[[T13:.*]] = llvm.extractelement %[[A]]{{\[}}%[[T12]] : i64] : vector<2xf32>
// CHECK:       %[[T14Insert:.*]] = llvm.insertelement %[[T13]]
// CHECK:       %[[T14:.*]] = llvm.shufflevector %[[T14Insert]]
// CHECK:       %[[T16:.*]] = llvm.extractvalue %[[T7]][1] : !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T17:.*]] = "llvm.intr.fmuladd"(%[[T14]], %[[B]], %[[T16]]) : (vector<3xf32>, vector<3xf32>, vector<3xf32>) -> vector<3xf32>
// CHECK:       %[[T18:.*]] = llvm.insertvalue %[[T17]], %[[T11]][1] : !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T19:.*]] = builtin.unrealized_conversion_cast %[[T18]] : !llvm.array<2 x vector<3xf32>> to vector<2x3xf32>
// CHECK:       return %[[T19]] : vector<2x3xf32>

// -----

func @shuffle_1D_direct(%arg0: vector<2xf32>, %arg1: vector<2xf32>) -> vector<2xf32> {
  %1 = vector.shuffle %arg0, %arg1 [0, 1] : vector<2xf32>, vector<2xf32>
  return %1 : vector<2xf32>
}
// CHECK-LABEL: @shuffle_1D_direct(
// CHECK-SAME: %[[A:.*]]: vector<2xf32>,
// CHECK-SAME: %[[B:.*]]: vector<2xf32>)
//       CHECK:   %[[s:.*]] = llvm.shufflevector %[[A]], %[[B]] [0, 1] : vector<2xf32>, vector<2xf32>
//       CHECK:   return %[[s]] : vector<2xf32>

// -----

func @shuffle_1D_index_direct(%arg0: vector<2xindex>, %arg1: vector<2xindex>) -> vector<2xindex> {
  %1 = vector.shuffle %arg0, %arg1 [0, 1] : vector<2xindex>, vector<2xindex>
  return %1 : vector<2xindex>
}
// CHECK-LABEL: @shuffle_1D_index_direct(
// CHECK-SAME: %[[A:.*]]: vector<2xindex>,
// CHECK-SAME: %[[B:.*]]: vector<2xindex>)
//       CHECK-DAG:   %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<2xindex> to vector<2xi64>
//       CHECK-DAG:   %[[T1:.*]] = builtin.unrealized_conversion_cast %[[B]] : vector<2xindex> to vector<2xi64>
//       CHECK:   %[[T2:.*]] = llvm.shufflevector %[[T0]], %[[T1]] [0, 1] : vector<2xi64>, vector<2xi64>
//       CHECK:   %[[T3:.*]] = builtin.unrealized_conversion_cast %[[T2]] : vector<2xi64> to vector<2xindex>
//       CHECK:   return %[[T3]] : vector<2xindex>

// -----

func @shuffle_1D(%arg0: vector<2xf32>, %arg1: vector<3xf32>) -> vector<5xf32> {
  %1 = vector.shuffle %arg0, %arg1 [4, 3, 2, 1, 0] : vector<2xf32>, vector<3xf32>
  return %1 : vector<5xf32>
}
// CHECK-LABEL: @shuffle_1D(
// CHECK-SAME: %[[A:.*]]: vector<2xf32>,
// CHECK-SAME: %[[B:.*]]: vector<3xf32>)
//       CHECK:   %[[u0:.*]] = llvm.mlir.undef : vector<5xf32>
//       CHECK:   %[[c2:.*]] = llvm.mlir.constant(2 : index) : i64
//       CHECK:   %[[e1:.*]] = llvm.extractelement %[[B]][%[[c2]] : i64] : vector<3xf32>
//       CHECK:   %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
//       CHECK:   %[[i1:.*]] = llvm.insertelement %[[e1]], %[[u0]][%[[c0]] : i64] : vector<5xf32>
//       CHECK:   %[[c1:.*]] = llvm.mlir.constant(1 : index) : i64
//       CHECK:   %[[e2:.*]] = llvm.extractelement %[[B]][%[[c1]] : i64] : vector<3xf32>
//       CHECK:   %[[c1:.*]] = llvm.mlir.constant(1 : index) : i64
//       CHECK:   %[[i2:.*]] = llvm.insertelement %[[e2]], %[[i1]][%[[c1]] : i64] : vector<5xf32>
//       CHECK:   %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
//       CHECK:   %[[e3:.*]] = llvm.extractelement %[[B]][%[[c0]] : i64] : vector<3xf32>
//       CHECK:   %[[c2:.*]] = llvm.mlir.constant(2 : index) : i64
//       CHECK:   %[[i3:.*]] = llvm.insertelement %[[e3]], %[[i2]][%[[c2]] : i64] : vector<5xf32>
//       CHECK:   %[[c1:.*]] = llvm.mlir.constant(1 : index) : i64
//       CHECK:   %[[e4:.*]] = llvm.extractelement %[[A]][%[[c1]] : i64] : vector<2xf32>
//       CHECK:   %[[c3:.*]] = llvm.mlir.constant(3 : index) : i64
//       CHECK:   %[[i4:.*]] = llvm.insertelement %[[e4]], %[[i3]][%[[c3]] : i64] : vector<5xf32>
//       CHECK:   %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
//       CHECK:   %[[e5:.*]] = llvm.extractelement %[[A]][%[[c0]] : i64] : vector<2xf32>
//       CHECK:   %[[c4:.*]] = llvm.mlir.constant(4 : index) : i64
//       CHECK:   %[[i5:.*]] = llvm.insertelement %[[e5]], %[[i4]][%[[c4]] : i64] : vector<5xf32>
//       CHECK:   return %[[i5]] : vector<5xf32>

// -----

func @shuffle_2D(%a: vector<1x4xf32>, %b: vector<2x4xf32>) -> vector<3x4xf32> {
  %1 = vector.shuffle %a, %b[1, 0, 2] : vector<1x4xf32>, vector<2x4xf32>
  return %1 : vector<3x4xf32>
}
// CHECK-LABEL: @shuffle_2D(
// CHECK-SAME: %[[A:.*]]: vector<1x4xf32>,
// CHECK-SAME: %[[B:.*]]: vector<2x4xf32>)
//       CHECK-DAG:   %[[VAL_0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<1x4xf32> to !llvm.array<1 x vector<4xf32>>
//       CHECK-DAG:   %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[B]] : vector<2x4xf32> to !llvm.array<2 x vector<4xf32>>
//       CHECK:   %[[u0:.*]] = llvm.mlir.undef : !llvm.array<3 x vector<4xf32>>
//       CHECK:   %[[e1:.*]] = llvm.extractvalue %[[VAL_1]][0] : !llvm.array<2 x vector<4xf32>>
//       CHECK:   %[[i1:.*]] = llvm.insertvalue %[[e1]], %[[u0]][0] : !llvm.array<3 x vector<4xf32>>
//       CHECK:   %[[e2:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.array<1 x vector<4xf32>>
//       CHECK:   %[[i2:.*]] = llvm.insertvalue %[[e2]], %[[i1]][1] : !llvm.array<3 x vector<4xf32>>
//       CHECK:   %[[e3:.*]] = llvm.extractvalue %[[VAL_1]][1] : !llvm.array<2 x vector<4xf32>>
//       CHECK:   %[[i3:.*]] = llvm.insertvalue %[[e3]], %[[i2]][2] : !llvm.array<3 x vector<4xf32>>
//       CHECK:   %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[i3]] : !llvm.array<3 x vector<4xf32>> to vector<3x4xf32>
//       CHECK:   return %[[VAL_3]] : vector<3x4xf32>

// -----

// CHECK-LABEL: @extract_element_0d
func @extract_element_0d(%a: vector<f32>) -> f32 {
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.extractelement %{{.*}}[%[[C0]] : {{.*}}] : vector<1xf32>
  %1 = vector.extractelement %a[] : vector<f32>
  return %1 : f32
}

// -----

func @extract_element(%arg0: vector<16xf32>) -> f32 {
  %0 = arith.constant 15 : i32
  %1 = vector.extractelement %arg0[%0 : i32]: vector<16xf32>
  return %1 : f32
}
// CHECK-LABEL: @extract_element(
// CHECK-SAME: %[[A:.*]]: vector<16xf32>)
//       CHECK:   %[[c:.*]] = arith.constant 15 : i32
//       CHECK:   %[[x:.*]] = llvm.extractelement %[[A]][%[[c]] : i32] : vector<16xf32>
//       CHECK:   return %[[x]] : f32

// -----

func @extract_element_index(%arg0: vector<16xf32>) -> f32 {
  %0 = arith.constant 15 : index
  %1 = vector.extractelement %arg0[%0 : index]: vector<16xf32>
  return %1 : f32
}
// CHECK-LABEL: @extract_element_index(
// CHECK-SAME: %[[A:.*]]: vector<16xf32>)
//       CHECK:   %[[c:.*]] = arith.constant 15 : index
//       CHECK:   %[[i:.*]] = builtin.unrealized_conversion_cast %[[c]] : index to i64
//       CHECK:   %[[x:.*]] = llvm.extractelement %[[A]][%[[i]] : i64] : vector<16xf32>
//       CHECK:   return %[[x]] : f32

// -----

func @extract_element_from_vec_1d(%arg0: vector<16xf32>) -> f32 {
  %0 = vector.extract %arg0[15]: vector<16xf32>
  return %0 : f32
}
// CHECK-LABEL: @extract_element_from_vec_1d
//       CHECK:   llvm.mlir.constant(15 : i64) : i64
//       CHECK:   llvm.extractelement {{.*}}[{{.*}} : i64] : vector<16xf32>
//       CHECK:   return {{.*}} : f32

// -----

func @extract_index_element_from_vec_1d(%arg0: vector<16xindex>) -> index {
  %0 = vector.extract %arg0[15]: vector<16xindex>
  return %0 : index
}
// CHECK-LABEL: @extract_index_element_from_vec_1d(
// CHECK-SAME: %[[A:.*]]: vector<16xindex>)
//       CHECK:   %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<16xindex> to vector<16xi64>
//       CHECK:   %[[T1:.*]] = llvm.mlir.constant(15 : i64) : i64
//       CHECK:   %[[T2:.*]] = llvm.extractelement %[[T0]][%[[T1]] : i64] : vector<16xi64>
//       CHECK:   %[[T3:.*]] = builtin.unrealized_conversion_cast %[[T2]] : i64 to index
//       CHECK:   return %[[T3]] : index

// -----

func @extract_vec_2d_from_vec_3d(%arg0: vector<4x3x16xf32>) -> vector<3x16xf32> {
  %0 = vector.extract %arg0[0]: vector<4x3x16xf32>
  return %0 : vector<3x16xf32>
}
// CHECK-LABEL: @extract_vec_2d_from_vec_3d
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm.array<4 x array<3 x vector<16xf32>>>
//       CHECK:   return {{.*}} : vector<3x16xf32>

// -----

func @extract_vec_1d_from_vec_3d(%arg0: vector<4x3x16xf32>) -> vector<16xf32> {
  %0 = vector.extract %arg0[0, 0]: vector<4x3x16xf32>
  return %0 : vector<16xf32>
}
// CHECK-LABEL: @extract_vec_1d_from_vec_3d
//       CHECK:   llvm.extractvalue {{.*}}[0, 0] : !llvm.array<4 x array<3 x vector<16xf32>>>
//       CHECK:   return {{.*}} : vector<16xf32>

// -----

func @extract_element_from_vec_3d(%arg0: vector<4x3x16xf32>) -> f32 {
  %0 = vector.extract %arg0[0, 0, 0]: vector<4x3x16xf32>
  return %0 : f32
}
// CHECK-LABEL: @extract_element_from_vec_3d
//       CHECK:   llvm.extractvalue {{.*}}[0, 0] : !llvm.array<4 x array<3 x vector<16xf32>>>
//       CHECK:   llvm.mlir.constant(0 : i64) : i64
//       CHECK:   llvm.extractelement {{.*}}[{{.*}} : i64] : vector<16xf32>
//       CHECK:   return {{.*}} : f32

// -----

// CHECK-LABEL: @insert_element_0d
// CHECK-SAME: %[[A:.*]]: f32,
func @insert_element_0d(%a: f32, %b: vector<f32>) -> vector<f32> {
  // CHECK: %[[B:.*]] =  builtin.unrealized_conversion_cast %{{.*}} :
  // CHECK:   vector<f32> to vector<1xf32>
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: %[[x:.*]] = llvm.insertelement %[[A]], %[[B]][%[[C0]] : {{.*}}] : vector<1xf32>
  %1 = vector.insertelement %a, %b[] : vector<f32>
  return %1 : vector<f32>
}

// -----

func @insert_element(%arg0: f32, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = arith.constant 3 : i32
  %1 = vector.insertelement %arg0, %arg1[%0 : i32] : vector<4xf32>
  return %1 : vector<4xf32>
}
// CHECK-LABEL: @insert_element(
// CHECK-SAME: %[[A:.*]]: f32,
// CHECK-SAME: %[[B:.*]]: vector<4xf32>)
//       CHECK:   %[[c:.*]] = arith.constant 3 : i32
//       CHECK:   %[[x:.*]] = llvm.insertelement %[[A]], %[[B]][%[[c]] : i32] : vector<4xf32>
//       CHECK:   return %[[x]] : vector<4xf32>

// -----

func @insert_element_index(%arg0: f32, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = arith.constant 3 : index
  %1 = vector.insertelement %arg0, %arg1[%0 : index] : vector<4xf32>
  return %1 : vector<4xf32>
}
// CHECK-LABEL: @insert_element_index(
//  CHECK-SAME: %[[A:.*]]: f32,
//  CHECK-SAME: %[[B:.*]]: vector<4xf32>)
//       CHECK:   %[[c:.*]] = arith.constant 3 : index
//       CHECK:   %[[i:.*]] = builtin.unrealized_conversion_cast %[[c]] : index to i64
//       CHECK:   %[[x:.*]] = llvm.insertelement %[[A]], %[[B]][%[[i]] : i64] : vector<4xf32>
//       CHECK:   return %[[x]] : vector<4xf32>

// -----

func @insert_element_into_vec_1d(%arg0: f32, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.insert %arg0, %arg1[3] : f32 into vector<4xf32>
  return %0 : vector<4xf32>
}
// CHECK-LABEL: @insert_element_into_vec_1d
//       CHECK:   llvm.mlir.constant(3 : i64) : i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : i64] : vector<4xf32>
//       CHECK:   return {{.*}} : vector<4xf32>

// -----

func @insert_index_element_into_vec_1d(%arg0: index, %arg1: vector<4xindex>) -> vector<4xindex> {
  %0 = vector.insert %arg0, %arg1[3] : index into vector<4xindex>
  return %0 : vector<4xindex>
}
// CHECK-LABEL: @insert_index_element_into_vec_1d(
// CHECK-SAME: %[[A:.*]]: index,
// CHECK-SAME: %[[B:.*]]: vector<4xindex>)
//       CHECK-DAG:   %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : index to i64
//       CHECK-DAG:   %[[T1:.*]] = builtin.unrealized_conversion_cast %[[B]] : vector<4xindex> to vector<4xi64>
//       CHECK:   %[[T3:.*]] = llvm.mlir.constant(3 : i64) : i64
//       CHECK:   %[[T4:.*]] = llvm.insertelement %[[T0]], %[[T1]][%[[T3]] : i64] : vector<4xi64>
//       CHECK:   %[[T5:.*]] = builtin.unrealized_conversion_cast %[[T4]] : vector<4xi64> to vector<4xindex>
//       CHECK:   return %[[T5]] : vector<4xindex>

// -----

func @insert_vec_2d_into_vec_3d(%arg0: vector<8x16xf32>, %arg1: vector<4x8x16xf32>) -> vector<4x8x16xf32> {
  %0 = vector.insert %arg0, %arg1[3] : vector<8x16xf32> into vector<4x8x16xf32>
  return %0 : vector<4x8x16xf32>
}
// CHECK-LABEL: @insert_vec_2d_into_vec_3d
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3] : !llvm.array<4 x array<8 x vector<16xf32>>>
//       CHECK:   return {{.*}} : vector<4x8x16xf32>

// -----

func @insert_vec_1d_into_vec_3d(%arg0: vector<16xf32>, %arg1: vector<4x8x16xf32>) -> vector<4x8x16xf32> {
  %0 = vector.insert %arg0, %arg1[3, 7] : vector<16xf32> into vector<4x8x16xf32>
  return %0 : vector<4x8x16xf32>
}
// CHECK-LABEL: @insert_vec_1d_into_vec_3d
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3, 7] : !llvm.array<4 x array<8 x vector<16xf32>>>
//       CHECK:   return {{.*}} : vector<4x8x16xf32>

// -----

func @insert_element_into_vec_3d(%arg0: f32, %arg1: vector<4x8x16xf32>) -> vector<4x8x16xf32> {
  %0 = vector.insert %arg0, %arg1[3, 7, 15] : f32 into vector<4x8x16xf32>
  return %0 : vector<4x8x16xf32>
}
// CHECK-LABEL: @insert_element_into_vec_3d
//       CHECK:   llvm.extractvalue {{.*}}[3, 7] : !llvm.array<4 x array<8 x vector<16xf32>>>
//       CHECK:   llvm.mlir.constant(15 : i64) : i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : i64] : vector<16xf32>
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3, 7] : !llvm.array<4 x array<8 x vector<16xf32>>>
//       CHECK:   return {{.*}} : vector<4x8x16xf32>

// -----

func @vector_type_cast(%arg0: memref<8x8x8xf32>) -> memref<vector<8x8x8xf32>> {
  %0 = vector.type_cast %arg0: memref<8x8x8xf32> to memref<vector<8x8x8xf32>>
  return %0 : memref<vector<8x8x8xf32>>
}
// CHECK-LABEL: @vector_type_cast
//       CHECK:   llvm.mlir.undef : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>>, ptr<array<8 x array<8 x vector<8xf32>>>>, i64)>
//       CHECK:   %[[allocated:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   %[[allocatedBit:.*]] = llvm.bitcast %[[allocated]] : !llvm.ptr<f32> to !llvm.ptr<array<8 x array<8 x vector<8xf32>>>>
//       CHECK:   llvm.insertvalue %[[allocatedBit]], {{.*}}[0] : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>>, ptr<array<8 x array<8 x vector<8xf32>>>>, i64)>
//       CHECK:   %[[aligned:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   %[[alignedBit:.*]] = llvm.bitcast %[[aligned]] : !llvm.ptr<f32> to !llvm.ptr<array<8 x array<8 x vector<8xf32>>>>
//       CHECK:   llvm.insertvalue %[[alignedBit]], {{.*}}[1] : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>>, ptr<array<8 x array<8 x vector<8xf32>>>>, i64)>
//       CHECK:   llvm.mlir.constant(0 : index
//       CHECK:   llvm.insertvalue {{.*}}[2] : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>>, ptr<array<8 x array<8 x vector<8xf32>>>>, i64)>

// -----

func @vector_index_type_cast(%arg0: memref<8x8x8xindex>) -> memref<vector<8x8x8xindex>> {
  %0 = vector.type_cast %arg0: memref<8x8x8xindex> to memref<vector<8x8x8xindex>>
  return %0 : memref<vector<8x8x8xindex>>
}
// CHECK-LABEL: @vector_index_type_cast(
// CHECK-SAME: %[[A:.*]]: memref<8x8x8xindex>)
//       CHECK:   %{{.*}} = builtin.unrealized_conversion_cast %[[A]] : memref<8x8x8xindex> to !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>

//       CHECK:   %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<(ptr<array<8 x array<8 x vector<8xi64>>>>, ptr<array<8 x array<8 x vector<8xi64>>>>, i64)> to memref<vector<8x8x8xindex>>

// -----

func @vector_type_cast_non_zero_addrspace(%arg0: memref<8x8x8xf32, 3>) -> memref<vector<8x8x8xf32>, 3> {
  %0 = vector.type_cast %arg0: memref<8x8x8xf32, 3> to memref<vector<8x8x8xf32>, 3>
  return %0 : memref<vector<8x8x8xf32>, 3>
}
// CHECK-LABEL: @vector_type_cast_non_zero_addrspace
//       CHECK:   llvm.mlir.undef : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>, 3>, ptr<array<8 x array<8 x vector<8xf32>>>, 3>, i64)>
//       CHECK:   %[[allocated:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   %[[allocatedBit:.*]] = llvm.bitcast %[[allocated]] : !llvm.ptr<f32, 3> to !llvm.ptr<array<8 x array<8 x vector<8xf32>>>, 3>
//       CHECK:   llvm.insertvalue %[[allocatedBit]], {{.*}}[0] : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>, 3>, ptr<array<8 x array<8 x vector<8xf32>>>, 3>, i64)>
//       CHECK:   %[[aligned:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   %[[alignedBit:.*]] = llvm.bitcast %[[aligned]] : !llvm.ptr<f32, 3> to !llvm.ptr<array<8 x array<8 x vector<8xf32>>>, 3>
//       CHECK:   llvm.insertvalue %[[alignedBit]], {{.*}}[1] : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>, 3>, ptr<array<8 x array<8 x vector<8xf32>>>, 3>, i64)>
//       CHECK:   llvm.mlir.constant(0 : index
//       CHECK:   llvm.insertvalue {{.*}}[2] : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>, 3>, ptr<array<8 x array<8 x vector<8xf32>>>, 3>, i64)>

// -----

func @vector_print_scalar_i1(%arg0: i1) {
  vector.print %arg0 : i1
  return
}
//
// Type "boolean" always uses zero extension.
//
// CHECK-LABEL: @vector_print_scalar_i1(
// CHECK-SAME: %[[A:.*]]: i1)
//       CHECK: %[[S:.*]] = arith.extui %[[A]] : i1 to i64
//       CHECK: llvm.call @printI64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

func @vector_print_scalar_i4(%arg0: i4) {
  vector.print %arg0 : i4
  return
}
// CHECK-LABEL: @vector_print_scalar_i4(
// CHECK-SAME: %[[A:.*]]: i4)
//       CHECK: %[[S:.*]] = arith.extsi %[[A]] : i4 to i64
//       CHECK: llvm.call @printI64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

func @vector_print_scalar_si4(%arg0: si4) {
  vector.print %arg0 : si4
  return
}
// CHECK-LABEL: @vector_print_scalar_si4(
// CHECK-SAME: %[[A:.*]]: si4)
//       CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[A]] : si4 to i4
//       CHECK: %[[S:.*]] = arith.extsi %[[C]] : i4 to i64
//       CHECK: llvm.call @printI64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

func @vector_print_scalar_ui4(%arg0: ui4) {
  vector.print %arg0 : ui4
  return
}
// CHECK-LABEL: @vector_print_scalar_ui4(
// CHECK-SAME: %[[A:.*]]: ui4)
//       CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[A]] : ui4 to i4
//       CHECK: %[[S:.*]] = arith.extui %[[C]] : i4 to i64
//       CHECK: llvm.call @printU64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

func @vector_print_scalar_i32(%arg0: i32) {
  vector.print %arg0 : i32
  return
}
// CHECK-LABEL: @vector_print_scalar_i32(
// CHECK-SAME: %[[A:.*]]: i32)
//       CHECK: %[[S:.*]] = arith.extsi %[[A]] : i32 to i64
//       CHECK: llvm.call @printI64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

func @vector_print_scalar_ui32(%arg0: ui32) {
  vector.print %arg0 : ui32
  return
}
// CHECK-LABEL: @vector_print_scalar_ui32(
// CHECK-SAME: %[[A:.*]]: ui32)
//       CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[A]] : ui32 to i32
//       CHECK: %[[S:.*]] = arith.extui %[[C]] : i32 to i64
//       CHECK: llvm.call @printU64(%[[S]]) : (i64) -> ()

// -----

func @vector_print_scalar_i40(%arg0: i40) {
  vector.print %arg0 : i40
  return
}
// CHECK-LABEL: @vector_print_scalar_i40(
// CHECK-SAME: %[[A:.*]]: i40)
//       CHECK: %[[S:.*]] = arith.extsi %[[A]] : i40 to i64
//       CHECK: llvm.call @printI64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

func @vector_print_scalar_si40(%arg0: si40) {
  vector.print %arg0 : si40
  return
}
// CHECK-LABEL: @vector_print_scalar_si40(
// CHECK-SAME: %[[A:.*]]: si40)
//       CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[A]] : si40 to i40
//       CHECK: %[[S:.*]] = arith.extsi %[[C]] : i40 to i64
//       CHECK: llvm.call @printI64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

func @vector_print_scalar_ui40(%arg0: ui40) {
  vector.print %arg0 : ui40
  return
}
// CHECK-LABEL: @vector_print_scalar_ui40(
// CHECK-SAME: %[[A:.*]]: ui40)
//       CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[A]] : ui40 to i40
//       CHECK: %[[S:.*]] = arith.extui %[[C]] : i40 to i64
//       CHECK: llvm.call @printU64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

func @vector_print_scalar_i64(%arg0: i64) {
  vector.print %arg0 : i64
  return
}
// CHECK-LABEL: @vector_print_scalar_i64(
// CHECK-SAME: %[[A:.*]]: i64)
//       CHECK:    llvm.call @printI64(%[[A]]) : (i64) -> ()
//       CHECK:    llvm.call @printNewline() : () -> ()

// -----

func @vector_print_scalar_ui64(%arg0: ui64) {
  vector.print %arg0 : ui64
  return
}
// CHECK-LABEL: @vector_print_scalar_ui64(
// CHECK-SAME: %[[A:.*]]: ui64)
//       CHECK:    %[[C:.*]] = builtin.unrealized_conversion_cast %[[A]] : ui64 to i64
//       CHECK:    llvm.call @printU64(%[[C]]) : (i64) -> ()
//       CHECK:    llvm.call @printNewline() : () -> ()

// -----

func @vector_print_scalar_index(%arg0: index) {
  vector.print %arg0 : index
  return
}
// CHECK-LABEL: @vector_print_scalar_index(
// CHECK-SAME: %[[A:.*]]: index)
//       CHECK:    %[[C:.*]] = builtin.unrealized_conversion_cast %[[A]] : index to i64
//       CHECK:    llvm.call @printU64(%[[C]]) : (i64) -> ()
//       CHECK:    llvm.call @printNewline() : () -> ()

// -----

func @vector_print_scalar_f32(%arg0: f32) {
  vector.print %arg0 : f32
  return
}
// CHECK-LABEL: @vector_print_scalar_f32(
// CHECK-SAME: %[[A:.*]]: f32)
//       CHECK:    llvm.call @printF32(%[[A]]) : (f32) -> ()
//       CHECK:    llvm.call @printNewline() : () -> ()

// -----

func @vector_print_scalar_f64(%arg0: f64) {
  vector.print %arg0 : f64
  return
}
// CHECK-LABEL: @vector_print_scalar_f64(
// CHECK-SAME: %[[A:.*]]: f64)
//       CHECK:    llvm.call @printF64(%[[A]]) : (f64) -> ()
//       CHECK:    llvm.call @printNewline() : () -> ()

// -----

func @vector_print_vector_0d(%arg0: vector<f32>) {
  vector.print %arg0 : vector<f32>
  return
}
// CHECK-LABEL: @vector_print_vector_0d(
// CHECK-SAME: %[[A:.*]]: vector<f32>)
//       CHECK: %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<f32> to vector<1xf32>
//       CHECK: llvm.call @printOpen() : () -> ()
//       CHECK: %[[T1:.*]] = llvm.mlir.constant(0 : index) : i64
//       CHECK: %[[T2:.*]] = llvm.extractelement %[[T0]][%[[T1]] : i64] : vector<1xf32>
//       CHECK: llvm.call @printF32(%[[T2]]) : (f32) -> ()
//       CHECK: llvm.call @printClose() : () -> ()
//       CHECK: llvm.call @printNewline() : () -> ()
//       CHECK: return

// -----

func @vector_print_vector(%arg0: vector<2x2xf32>) {
  vector.print %arg0 : vector<2x2xf32>
  return
}
// CHECK-LABEL: @vector_print_vector(
// CHECK-SAME: %[[A:.*]]: vector<2x2xf32>)
//       CHECK:    %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<2x2xf32> to !llvm.array<2 x vector<2xf32>>
//       CHECK:    llvm.call @printOpen() : () -> ()
//       CHECK:    %[[x0:.*]] = llvm.extractvalue %[[VAL_1]][0] : !llvm.array<2 x vector<2xf32>>
//       CHECK:    llvm.call @printOpen() : () -> ()
//       CHECK:    %[[x1:.*]] = llvm.mlir.constant(0 : index) : i64
//       CHECK:    %[[x2:.*]] = llvm.extractelement %[[x0]][%[[x1]] : i64] : vector<2xf32>
//       CHECK:    llvm.call @printF32(%[[x2]]) : (f32) -> ()
//       CHECK:    llvm.call @printComma() : () -> ()
//       CHECK:    %[[x3:.*]] = llvm.mlir.constant(1 : index) : i64
//       CHECK:    %[[x4:.*]] = llvm.extractelement %[[x0]][%[[x3]] : i64] : vector<2xf32>
//       CHECK:    llvm.call @printF32(%[[x4]]) : (f32) -> ()
//       CHECK:    llvm.call @printClose() : () -> ()
//       CHECK:    llvm.call @printComma() : () -> ()
//       CHECK:    %[[x5:.*]] = llvm.extractvalue %[[VAL_1]][1] : !llvm.array<2 x vector<2xf32>>
//       CHECK:    llvm.call @printOpen() : () -> ()
//       CHECK:    %[[x6:.*]] = llvm.mlir.constant(0 : index) : i64
//       CHECK:    %[[x7:.*]] = llvm.extractelement %[[x5]][%[[x6]] : i64] : vector<2xf32>
//       CHECK:    llvm.call @printF32(%[[x7]]) : (f32) -> ()
//       CHECK:    llvm.call @printComma() : () -> ()
//       CHECK:    %[[x8:.*]] = llvm.mlir.constant(1 : index) : i64
//       CHECK:    %[[x9:.*]] = llvm.extractelement %[[x5]][%[[x8]] : i64] : vector<2xf32>
//       CHECK:    llvm.call @printF32(%[[x9]]) : (f32) -> ()
//       CHECK:    llvm.call @printClose() : () -> ()
//       CHECK:    llvm.call @printClose() : () -> ()
//       CHECK:    llvm.call @printNewline() : () -> ()

// -----

func @extract_strided_slice1(%arg0: vector<4xf32>) -> vector<2xf32> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
  return %0 : vector<2xf32>
}
// CHECK-LABEL: @extract_strided_slice1(
//  CHECK-SAME:    %[[A:.*]]: vector<4xf32>)
//       CHECK:    %[[T0:.*]] = llvm.shufflevector %[[A]], %[[A]] [2, 3] : vector<4xf32>, vector<4xf32>
//       CHECK:    return %[[T0]] : vector<2xf32>

// -----

func @extract_strided_index_slice1(%arg0: vector<4xindex>) -> vector<2xindex> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]} : vector<4xindex> to vector<2xindex>
  return %0 : vector<2xindex>
}
// CHECK-LABEL: @extract_strided_index_slice1(
//  CHECK-SAME:    %[[A:.*]]: vector<4xindex>)
//       CHECK:    %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<4xindex> to vector<4xi64>
//       CHECK:    %[[T2:.*]] = llvm.shufflevector %[[T0]], %[[T0]] [2, 3] : vector<4xi64>, vector<4xi64>
//       CHECK:    %[[T3:.*]] = builtin.unrealized_conversion_cast %[[T2]] : vector<2xi64> to vector<2xindex>
//       CHECK:    return %[[T3]] : vector<2xindex>

// -----

func @extract_strided_slice2(%arg0: vector<4x8xf32>) -> vector<2x8xf32> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]} : vector<4x8xf32> to vector<2x8xf32>
  return %0 : vector<2x8xf32>
}
// CHECK-LABEL: @extract_strided_slice2(
//  CHECK-SAME:    %[[ARG:.*]]: vector<4x8xf32>)
//       CHECK:    %[[A:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : vector<4x8xf32> to !llvm.array<4 x vector<8xf32>>
//       CHECK:    %[[T0:.*]] = llvm.mlir.undef : !llvm.array<2 x vector<8xf32>>
//       CHECK:    %[[T1:.*]] = llvm.extractvalue %[[A]][2] : !llvm.array<4 x vector<8xf32>>
//       CHECK:    %[[T2:.*]] = llvm.insertvalue %[[T1]], %[[T0]][0] : !llvm.array<2 x vector<8xf32>>
//       CHECK:    %[[T3:.*]] = llvm.extractvalue %[[A]][3] : !llvm.array<4 x vector<8xf32>>
//       CHECK:    %[[T4:.*]] = llvm.insertvalue %[[T3]], %[[T2]][1] : !llvm.array<2 x vector<8xf32>>
//       CHECK:    %[[T5:.*]] = builtin.unrealized_conversion_cast %[[T4]] : !llvm.array<2 x vector<8xf32>> to vector<2x8xf32>
//       CHECK:    return %[[T5]]

// -----

func @extract_strided_slice3(%arg0: vector<4x8xf32>) -> vector<2x2xf32> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x8xf32> to vector<2x2xf32>
  return %0 : vector<2x2xf32>
}
// CHECK-LABEL: @extract_strided_slice3(
//  CHECK-SAME:    %[[ARG:.*]]: vector<4x8xf32>)
//       CHECK:    %[[A:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : vector<4x8xf32> to !llvm.array<4 x vector<8xf32>>
//       CHECK:    %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : vector<2x2xf32>
//       CHECK:    %[[VAL_6:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : vector<2x2xf32> to !llvm.array<2 x vector<2xf32>>
//       CHECK:    %[[T2:.*]] = llvm.extractvalue %[[A]][2] : !llvm.array<4 x vector<8xf32>>
//       CHECK:    %[[T3:.*]] = llvm.shufflevector %[[T2]], %[[T2]] [2, 3] : vector<8xf32>, vector<8xf32>
//       CHECK:    %[[T4:.*]] = llvm.insertvalue %[[T3]], %[[VAL_6]][0] : !llvm.array<2 x vector<2xf32>>
//       CHECK:    %[[T5:.*]] = llvm.extractvalue %[[A]][3] : !llvm.array<4 x vector<8xf32>>
//       CHECK:    %[[T6:.*]] = llvm.shufflevector %[[T5]], %[[T5]] [2, 3] : vector<8xf32>, vector<8xf32>
//       CHECK:    %[[T7:.*]] = llvm.insertvalue %[[T6]], %[[T4]][1] : !llvm.array<2 x vector<2xf32>>
//       CHECK:    %[[VAL_12:.*]] = builtin.unrealized_conversion_cast %[[T7]] : !llvm.array<2 x vector<2xf32>> to vector<2x2xf32>
//       CHECK:    return %[[VAL_12]] : vector<2x2xf32>

// -----

func @insert_strided_slice1(%b: vector<4x4xf32>, %c: vector<4x4x4xf32>) -> vector<4x4x4xf32> {
  %0 = vector.insert_strided_slice %b, %c {offsets = [2, 0, 0], strides = [1, 1]} : vector<4x4xf32> into vector<4x4x4xf32>
  return %0 : vector<4x4x4xf32>
}
// CHECK-LABEL: @insert_strided_slice1
//       CHECK:    llvm.extractvalue {{.*}}[2] : !llvm.array<4 x array<4 x vector<4xf32>>>
//       CHECK:    llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.array<4 x array<4 x vector<4xf32>>>

// -----

func @insert_strided_index_slice1(%b: vector<4x4xindex>, %c: vector<4x4x4xindex>) -> vector<4x4x4xindex> {
  %0 = vector.insert_strided_slice %b, %c {offsets = [2, 0, 0], strides = [1, 1]} : vector<4x4xindex> into vector<4x4x4xindex>
  return %0 : vector<4x4x4xindex>
}
// CHECK-LABEL: @insert_strided_index_slice1(
//       CHECK:    llvm.extractvalue {{.*}}[2] : !llvm.array<4 x array<4 x vector<4xi64>>>
//       CHECK:    llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.array<4 x array<4 x vector<4xi64>>>

// -----

func @insert_strided_slice2(%a: vector<2x2xf32>, %b: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: @insert_strided_slice2
//
// Subvector vector<2xf32> @0 into vector<4xf32> @2
//       CHECK:    %[[V2_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.array<2 x vector<2xf32>>
//       CHECK:    %[[V4_0:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.array<4 x vector<4xf32>>
// Element @0 -> element @2
//       CHECK:    %[[R4_0:.*]] = llvm.shufflevector %[[V2_0]], %[[V2_0]] [0, 1, 0, 0] : vector<2xf32>, vector<2xf32>
//       CHECK:    %[[R4_1:.*]] = llvm.shufflevector %[[R4_0]], %[[V4_0]] [4, 5, 0, 1] : vector<4xf32>, vector<4xf32>
//       CHECK:    llvm.insertvalue %[[R4_1]], {{.*}}[2] : !llvm.array<4 x vector<4xf32>>
//
// Subvector vector<2xf32> @1 into vector<4xf32> @3
//       CHECK:    %[[V2_1:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.array<2 x vector<2xf32>>
//       CHECK:    %[[V4_3:.*]] = llvm.extractvalue {{.*}}[3] : !llvm.array<4 x vector<4xf32>>
// Element @0 -> element @2
//       CHECK:    %[[R4_2:.*]] = llvm.shufflevector %[[V2_1]], %[[V2_1]] [0, 1, 0, 0] : vector<2xf32>, vector<2xf32>
//       CHECK:    %[[R4_3:.*]] = llvm.shufflevector %[[R4_2]], %[[V4_3]] [4, 5, 0, 1] : vector<4xf32>, vector<4xf32>
//       CHECK:    llvm.insertvalue %[[R4_3]], {{.*}}[3] : !llvm.array<4 x vector<4xf32>>

// -----

func @insert_strided_slice3(%arg0: vector<2x4xf32>, %arg1: vector<16x4x8xf32>) -> vector<16x4x8xf32> {
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [0, 0, 2], strides = [1, 1]}:
        vector<2x4xf32> into vector<16x4x8xf32>
  return %0 : vector<16x4x8xf32>
}
// CHECK-LABEL: func @insert_strided_slice3
//       CHECK:    %[[V4_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.array<2 x vector<4xf32>>
//       CHECK:    %[[V4_0_0:.*]] = llvm.extractvalue {{.*}}[0, 0] : !llvm.array<16 x array<4 x vector<8xf32>>>
//       CHECK:    %[[R8_0:.*]] = llvm.shufflevector %[[V4_0]], %[[V4_0]] [0, 1, 2, 3, 0, 0, 0, 0] : vector<4xf32>, vector<4xf32>
//       CHECK:    %[[R8_1:.*]] = llvm.shufflevector %[[R8_0:.*]], %[[V4_0_0]] [8, 9, 0, 1, 2, 3, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:    llvm.insertvalue %[[R8_1]], {{.*}}[0] : !llvm.array<4 x vector<8xf32>>

//       CHECK:    %[[V4_1:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.array<2 x vector<4xf32>>
//       CHECK:    %[[V4_0_1:.*]] = llvm.extractvalue {{.*}}[0, 1] : !llvm.array<16 x array<4 x vector<8xf32>>>
//       CHECK:    %[[R8_2:.*]] = llvm.shufflevector %[[V4_1]], %[[V4_1]] [0, 1, 2, 3, 0, 0, 0, 0] : vector<4xf32>, vector<4xf32>
//       CHECK:    %[[R8_3:.*]] = llvm.shufflevector %[[R8_2]], %[[V4_0_1]] [8, 9, 0, 1, 2, 3, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:    llvm.insertvalue %[[R8_3]], {{.*}}[1] : !llvm.array<4 x vector<8xf32>>

// -----

func @vector_fma(%a: vector<8xf32>, %b: vector<2x4xf32>, %c: vector<1x1x1xf32>) -> (vector<8xf32>, vector<2x4xf32>, vector<1x1x1xf32>) {
  // CHECK-LABEL: @vector_fma
  //  CHECK-SAME: %[[A:.*]]: vector<8xf32>
  //  CHECK-SAME: %[[B:.*]]: vector<2x4xf32>
  //  CHECK-SAME: %[[C:.*]]: vector<1x1x1xf32>
  //       CHECK: %[[BL:.*]] = builtin.unrealized_conversion_cast %[[B]] : vector<2x4xf32> to !llvm.array<2 x vector<4xf32>>
  //       CHECK: "llvm.intr.fmuladd"
  //  CHECK-SAME:   (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  %0 = vector.fma %a, %a, %a : vector<8xf32>

  //       CHECK: %[[b00:.*]] = llvm.extractvalue %[[BL]][0] : !llvm.array<2 x vector<4xf32>>
  //       CHECK: %[[b01:.*]] = llvm.extractvalue %[[BL]][0] : !llvm.array<2 x vector<4xf32>>
  //       CHECK: %[[b02:.*]] = llvm.extractvalue %[[BL]][0] : !llvm.array<2 x vector<4xf32>>
  //       CHECK: %[[B0:.*]] = "llvm.intr.fmuladd"(%[[b00]], %[[b01]], %[[b02]]) :
  //  CHECK-SAME: (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  //       CHECK: llvm.insertvalue %[[B0]], {{.*}}[0] : !llvm.array<2 x vector<4xf32>>
  //       CHECK: %[[b10:.*]] = llvm.extractvalue %[[BL]][1] : !llvm.array<2 x vector<4xf32>>
  //       CHECK: %[[b11:.*]] = llvm.extractvalue %[[BL]][1] : !llvm.array<2 x vector<4xf32>>
  //       CHECK: %[[b12:.*]] = llvm.extractvalue %[[BL]][1] : !llvm.array<2 x vector<4xf32>>
  //       CHECK: %[[B1:.*]] = "llvm.intr.fmuladd"(%[[b10]], %[[b11]], %[[b12]]) :
  //  CHECK-SAME: (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  //       CHECK: llvm.insertvalue %[[B1]], {{.*}}[1] : !llvm.array<2 x vector<4xf32>>
  %1 = vector.fma %b, %b, %b : vector<2x4xf32>

  //       CHECK: %[[C0:.*]] = "llvm.intr.fmuladd"
  //  CHECK-SAME:   (vector<1xf32>, vector<1xf32>, vector<1xf32>) -> vector<1xf32>
  %2 = vector.fma %c, %c, %c : vector<1x1x1xf32>

  return %0, %1, %2: vector<8xf32>, vector<2x4xf32>, vector<1x1x1xf32>
}

// -----

func @reduce_f16(%arg0: vector<16xf16>) -> f16 {
  %0 = vector.reduction <add>, %arg0 : vector<16xf16> into f16
  return %0 : f16
}
// CHECK-LABEL: @reduce_f16(
// CHECK-SAME: %[[A:.*]]: vector<16xf16>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f16) : f16
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// CHECK-SAME: {reassoc = false} : (f16, vector<16xf16>) -> f16
//      CHECK: return %[[V]] : f16

// -----

func @reduce_f32(%arg0: vector<16xf32>) -> f32 {
  %0 = vector.reduction <add>, %arg0 : vector<16xf32> into f32
  return %0 : f32
}
// CHECK-LABEL: @reduce_f32(
// CHECK-SAME: %[[A:.*]]: vector<16xf32>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// CHECK-SAME: {reassoc = false} : (f32, vector<16xf32>) -> f32
//      CHECK: return %[[V]] : f32

// -----

func @reduce_f64(%arg0: vector<16xf64>) -> f64 {
  %0 = vector.reduction <add>, %arg0 : vector<16xf64> into f64
  return %0 : f64
}
// CHECK-LABEL: @reduce_f64(
// CHECK-SAME: %[[A:.*]]: vector<16xf64>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f64) : f64
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// CHECK-SAME: {reassoc = false} : (f64, vector<16xf64>) -> f64
//      CHECK: return %[[V]] : f64

// -----

func @reduce_i8(%arg0: vector<16xi8>) -> i8 {
  %0 = vector.reduction <add>, %arg0 : vector<16xi8> into i8
  return %0 : i8
}
// CHECK-LABEL: @reduce_i8(
// CHECK-SAME: %[[A:.*]]: vector<16xi8>)
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.add"(%[[A]])
//      CHECK: return %[[V]] : i8

// -----

func @reduce_i32(%arg0: vector<16xi32>) -> i32 {
  %0 = vector.reduction <add>, %arg0 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_i32(
// CHECK-SAME: %[[A:.*]]: vector<16xi32>)
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.add"(%[[A]])
//      CHECK: return %[[V]] : i32

// -----

func @reduce_i64(%arg0: vector<16xi64>) -> i64 {
  %0 = vector.reduction <add>, %arg0 : vector<16xi64> into i64
  return %0 : i64
}
// CHECK-LABEL: @reduce_i64(
// CHECK-SAME: %[[A:.*]]: vector<16xi64>)
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.add"(%[[A]])
//      CHECK: return %[[V]] : i64

// -----

func @reduce_index(%arg0: vector<16xindex>) -> index {
  %0 = vector.reduction <add>, %arg0 : vector<16xindex> into index
  return %0 : index
}
// CHECK-LABEL: @reduce_index(
// CHECK-SAME: %[[A:.*]]: vector<16xindex>)
//      CHECK: %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<16xindex> to vector<16xi64>
//      CHECK: %[[T1:.*]] = "llvm.intr.vector.reduce.add"(%[[T0]])
//      CHECK: %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T1]] : i64 to index
//      CHECK: return %[[T2]] : index

//                          4x16                16x3               4x3
// -----

func @matrix_ops(%A: vector<64xf64>, %B: vector<48xf64>) -> vector<12xf64> {
  %C = vector.matrix_multiply %A, %B
    { lhs_rows = 4: i32, lhs_columns = 16: i32 , rhs_columns = 3: i32 } :
    (vector<64xf64>, vector<48xf64>) -> vector<12xf64>
  return %C: vector<12xf64>
}
// CHECK-LABEL: @matrix_ops
//       CHECK:   llvm.intr.matrix.multiply %{{.*}}, %{{.*}} {
//  CHECK-SAME: lhs_columns = 16 : i32, lhs_rows = 4 : i32, rhs_columns = 3 : i32
//  CHECK-SAME: } : (vector<64xf64>, vector<48xf64>) -> vector<12xf64>

// -----

func @matrix_ops_index(%A: vector<64xindex>, %B: vector<48xindex>) -> vector<12xindex> {
  %C = vector.matrix_multiply %A, %B
    { lhs_rows = 4: i32, lhs_columns = 16: i32 , rhs_columns = 3: i32 } :
    (vector<64xindex>, vector<48xindex>) -> vector<12xindex>
  return %C: vector<12xindex>
}
// CHECK-LABEL: @matrix_ops_index
//       CHECK:   llvm.intr.matrix.multiply %{{.*}}, %{{.*}} {
//  CHECK-SAME: lhs_columns = 16 : i32, lhs_rows = 4 : i32, rhs_columns = 3 : i32
//  CHECK-SAME: } : (vector<64xi64>, vector<48xi64>) -> vector<12xi64>

// -----

func @transfer_read_1d(%A : memref<?xf32>, %base: index) -> vector<17xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32>, vector<17xf32>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<17xf32>, memref<?xf32>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_1d
//  CHECK-SAME: %[[BASE:[a-zA-Z0-9]*]]: index) -> vector<17xf32>
//       CHECK: %[[c7:.*]] = arith.constant 7.0
//       CHECK: %[[C0:.*]] = arith.constant 0 : index
//       CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %[[C0]] : memref<?xf32>
//
// 1. Create a vector with linear indices [ 0 .. vector_length - 1 ].
//       CHECK: %[[linearIndex:.*]] = arith.constant dense
//  CHECK-SAME: <[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> :
//  CHECK-SAME: vector<17xi32>
//
// 2. Create offsetVector = [ offset + 0 .. offset + vector_length - 1 ].
//       CHECK: %[[otrunc:.*]] = arith.index_cast %[[BASE]] : index to i32
//       CHECK: %[[offsetVecInsert:.*]] = llvm.insertelement %[[otrunc]]
//       CHECK: %[[offsetVec:.*]] = llvm.shufflevector %[[offsetVecInsert]]
//       CHECK: %[[offsetVec2:.*]] = arith.addi %[[offsetVec]], %[[linearIndex]] : vector<17xi32>
//
// 3. Let dim the memref dimension, compute the vector comparison mask:
//    [ offset + 0 .. offset + vector_length - 1 ] < [ dim .. dim ]
//       CHECK: %[[dtrunc:.*]] = arith.index_cast %[[DIM]] : index to i32
//       CHECK: %[[dimVecInsert:.*]] = llvm.insertelement %[[dtrunc]]
//       CHECK: %[[dimVec:.*]] = llvm.shufflevector %[[dimVecInsert]]
//       CHECK: %[[mask:.*]] = arith.cmpi slt, %[[offsetVec2]], %[[dimVec]] : vector<17xi32>
//
// 4. Create pass-through vector.
//       CHECK: %[[PASS_THROUGH:.*]] = arith.constant dense<7.{{.*}}> : vector<17xf32>
//
// 5. Bitcast to vector form.
//       CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
//       CHECK: %[[vecPtr:.*]] = llvm.bitcast %[[gep]] :
//  CHECK-SAME: !llvm.ptr<f32> to !llvm.ptr<vector<17xf32>>
//
// 6. Rewrite as a masked read.
//       CHECK: %[[loaded:.*]] = llvm.intr.masked.load %[[vecPtr]], %[[mask]],
//  CHECK-SAME: %[[PASS_THROUGH]] {alignment = 4 : i32} :
//  CHECK-SAME: (!llvm.ptr<vector<17xf32>>, vector<17xi1>, vector<17xf32>) -> vector<17xf32>
//
// 1. Create a vector with linear indices [ 0 .. vector_length - 1 ].
//       CHECK: %[[linearIndex_b:.*]] = arith.constant dense
//  CHECK-SAME: <[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> :
//  CHECK-SAME: vector<17xi32>
//
// 2. Create offsetVector = [ offset + 0 .. offset + vector_length - 1 ].
//       CHECK: llvm.shufflevector %{{.*}} : vector<17xi32>
//       CHECK: arith.addi
//
// 3. Let dim the memref dimension, compute the vector comparison mask:
//    [ offset + 0 .. offset + vector_length - 1 ] < [ dim .. dim ]
//       CHECK: llvm.shufflevector %{{.*}} : vector<17xi32>
//       CHECK: %[[mask_b:.*]] = arith.cmpi slt, {{.*}} : vector<17xi32>
//
// 4. Bitcast to vector form.
//       CHECK: %[[gep_b:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
//       CHECK: %[[vecPtr_b:.*]] = llvm.bitcast %[[gep_b]] :
//  CHECK-SAME: !llvm.ptr<f32> to !llvm.ptr<vector<17xf32>>
//
// 5. Rewrite as a masked write.
//       CHECK: llvm.intr.masked.store %[[loaded]], %[[vecPtr_b]], %[[mask_b]]
//  CHECK-SAME: {alignment = 4 : i32} :
//  CHECK-SAME: vector<17xf32>, vector<17xi1> into !llvm.ptr<vector<17xf32>>

// -----

func @transfer_read_index_1d(%A : memref<?xindex>, %base: index) -> vector<17xindex> {
  %f7 = arith.constant 7: index
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xindex>, vector<17xindex>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<17xindex>, memref<?xindex>
  return %f: vector<17xindex>
}
// CHECK-LABEL: func @transfer_read_index_1d
//  CHECK-SAME: %[[BASE:[a-zA-Z0-9]*]]: index) -> vector<17xindex>
//       CHECK: %[[SPLAT:.*]] = arith.constant dense<7> : vector<17xindex>
//       CHECK: %{{.*}} = builtin.unrealized_conversion_cast %[[SPLAT]] : vector<17xindex> to vector<17xi64>

//       CHECK: %[[loaded:.*]] = llvm.intr.masked.load %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} :
//  CHECK-SAME: (!llvm.ptr<vector<17xi64>>, vector<17xi1>, vector<17xi64>) -> vector<17xi64>

//       CHECK: llvm.intr.masked.store %[[loaded]], %{{.*}}, %{{.*}} {alignment = 8 : i32} :
//  CHECK-SAME: vector<17xi64>, vector<17xi1> into !llvm.ptr<vector<17xi64>>

// -----

func @transfer_read_2d_to_1d(%A : memref<?x?xf32>, %base0: index, %base1: index) -> vector<17xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base0, %base1], %f7
      {permutation_map = affine_map<(d0, d1) -> (d1)>} :
    memref<?x?xf32>, vector<17xf32>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_2d_to_1d
//  CHECK-SAME: %[[BASE_0:[a-zA-Z0-9]*]]: index, %[[BASE_1:[a-zA-Z0-9]*]]: index) -> vector<17xf32>
//       CHECK: %[[c1:.*]] = arith.constant 1 : index
//       CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %[[c1]] : memref<?x?xf32>
//
// Create offsetVector = [ offset + 0 .. offset + vector_length - 1 ].
//       CHECK: %[[trunc:.*]] = arith.index_cast %[[BASE_1]] : index to i32
//       CHECK: %[[offsetVecInsert:.*]] = llvm.insertelement %[[trunc]]
//       CHECK: %[[offsetVec:.*]] = llvm.shufflevector %[[offsetVecInsert]]
//
// Let dim the memref dimension, compute the vector comparison mask:
//    [ offset + 0 .. offset + vector_length - 1 ] < [ dim .. dim ]
//       CHECK: %[[dimtrunc:.*]] = arith.index_cast %[[DIM]] : index to i32
//       CHECK: %[[dimtruncInsert:.*]] = llvm.insertelement %[[dimtrunc]]
//       CHECK: llvm.shufflevector %[[dimtruncInsert]]

// -----

func @transfer_read_1d_non_zero_addrspace(%A : memref<?xf32, 3>, %base: index) -> vector<17xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32, 3>, vector<17xf32>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<17xf32>, memref<?xf32, 3>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_1d_non_zero_addrspace
//  CHECK-SAME: %[[BASE:[a-zA-Z0-9]*]]: index) -> vector<17xf32>
//
// 1. Check address space for GEP is correct.
//       CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm.ptr<f32, 3>, i64) -> !llvm.ptr<f32, 3>
//       CHECK: %[[vecPtr:.*]] = llvm.bitcast %[[gep]] :
//  CHECK-SAME: !llvm.ptr<f32, 3> to !llvm.ptr<vector<17xf32>, 3>
//
// 2. Check address space of the memref is correct.
//       CHECK: %[[c0:.*]] = arith.constant 0 : index
//       CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %[[c0]] : memref<?xf32, 3>
//
// 3. Check address space for GEP is correct.
//       CHECK: %[[gep_b:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm.ptr<f32, 3>, i64) -> !llvm.ptr<f32, 3>
//       CHECK: %[[vecPtr_b:.*]] = llvm.bitcast %[[gep_b]] :
//  CHECK-SAME: !llvm.ptr<f32, 3> to !llvm.ptr<vector<17xf32>, 3>

// -----

func @transfer_read_1d_inbounds(%A : memref<?xf32>, %base: index) -> vector<17xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7 {in_bounds = [true]} :
    memref<?xf32>, vector<17xf32>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_1d_inbounds
//  CHECK-SAME: %[[BASE:[a-zA-Z0-9]*]]: index) -> vector<17xf32>
//
// 1. Bitcast to vector form.
//       CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
//       CHECK: %[[vecPtr:.*]] = llvm.bitcast %[[gep]] :
//  CHECK-SAME: !llvm.ptr<f32> to !llvm.ptr<vector<17xf32>>
//
// 2. Rewrite as a load.
//       CHECK: %[[loaded:.*]] = llvm.load %[[vecPtr]] {alignment = 4 : i64} : !llvm.ptr<vector<17xf32>>

// -----

// CHECK-LABEL: func @transfer_read_1d_mask
// CHECK: %[[mask1:.*]] = arith.constant dense<[false, false, true, false, true]>
// CHECK: %[[cmpi:.*]] = arith.cmpi slt
// CHECK: %[[mask2:.*]] = arith.andi %[[cmpi]], %[[mask1]]
// CHECK: %[[r:.*]] = llvm.intr.masked.load %{{.*}}, %[[mask2]]
// CHECK: return %[[r]]
func @transfer_read_1d_mask(%A : memref<?xf32>, %base : index) -> vector<5xf32> {
  %m = arith.constant dense<[0, 0, 1, 0, 1]> : vector<5xi1>
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7, %m : memref<?xf32>, vector<5xf32>
  return %f: vector<5xf32>
}

// -----

func @genbool_0d_f() -> vector<i1> {
  %0 = vector.constant_mask [0] : vector<i1>
  return %0 : vector<i1>
}
// CHECK-LABEL: func @genbool_0d_f
// CHECK: %[[VAL_0:.*]] = arith.constant dense<false> : vector<i1>
// CHECK: return %[[VAL_0]] : vector<i1>

// -----

func @genbool_0d_t() -> vector<i1> {
  %0 = vector.constant_mask [1] : vector<i1>
  return %0 : vector<i1>
}
// CHECK-LABEL: func @genbool_0d_t
// CHECK: %[[VAL_0:.*]] = arith.constant dense<true> : vector<i1>
// CHECK: return %[[VAL_0]] : vector<i1>

// -----

func @genbool_1d() -> vector<8xi1> {
  %0 = vector.constant_mask [4] : vector<8xi1>
  return %0 : vector<8xi1>
}
// CHECK-LABEL: func @genbool_1d
// CHECK: %[[VAL_0:.*]] = arith.constant dense<[true, true, true, true, false, false, false, false]> : vector<8xi1>
// CHECK: return %[[VAL_0]] : vector<8xi1>

// -----

func @genbool_2d() -> vector<4x4xi1> {
  %v = vector.constant_mask [2, 2] : vector<4x4xi1>
  return %v: vector<4x4xi1>
}

// CHECK-LABEL: func @genbool_2d
// CHECK: %[[VAL_0:.*]] = arith.constant dense<[true, true, false, false]> : vector<4xi1>
// CHECK: %[[VAL_1:.*]] = arith.constant dense<false> : vector<4x4xi1>
// CHECK: %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : vector<4x4xi1> to !llvm.array<4 x vector<4xi1>>
// CHECK: %[[VAL_3:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_2]][0] : !llvm.array<4 x vector<4xi1>>
// CHECK: %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_3]][1] : !llvm.array<4 x vector<4xi1>>
// CHECK: %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_4]] : !llvm.array<4 x vector<4xi1>> to vector<4x4xi1>
// CHECK: return %[[VAL_5]] : vector<4x4xi1>

// -----

func @create_mask_0d(%a : index) -> vector<i1> {
  %v = vector.create_mask %a : vector<i1>
  return %v: vector<i1>
}

// CHECK-LABEL: func @create_mask_0d
// CHECK-SAME: %[[arg:.*]]: index
// CHECK:  %[[indices:.*]] = arith.constant dense<0> : vector<i32>
// CHECK:  %[[arg_i32:.*]] = arith.index_cast %[[arg]] : index to i32
// CHECK:  %[[bounds:.*]] = llvm.insertelement %[[arg_i32]]
// CHECK:  %[[boundsCast:.*]] = builtin.unrealized_conversion_cast %[[bounds]] : vector<1xi32> to vector<i32>
// CHECK:  %[[result:.*]] = arith.cmpi slt, %[[indices]], %[[boundsCast]] : vector<i32>
// CHECK:  return %[[result]] : vector<i1>

// -----

func @create_mask_1d(%a : index) -> vector<4xi1> {
  %v = vector.create_mask %a : vector<4xi1>
  return %v: vector<4xi1>
}

// CHECK-LABEL: func @create_mask_1d
// CHECK-SAME: %[[arg:.*]]: index
// CHECK:  %[[indices:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
// CHECK:  %[[arg_i32:.*]] = arith.index_cast %[[arg]] : index to i32
// CHECK:  %[[boundsInsert:.*]] = llvm.insertelement %[[arg_i32]]
// CHECK:  %[[bounds:.*]] = llvm.shufflevector %[[boundsInsert]]
// CHECK:  %[[result:.*]] = arith.cmpi slt, %[[indices]], %[[bounds]] : vector<4xi32>
// CHECK:  return %[[result]] : vector<4xi1>

// -----

func @flat_transpose(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = vector.flat_transpose %arg0 { rows = 4: i32, columns = 4: i32 }
     : vector<16xf32> -> vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @flat_transpose
// CHECK-SAME:  %[[A:.*]]: vector<16xf32>
// CHECK:       %[[T:.*]] = llvm.intr.matrix.transpose %[[A]]
// CHECK-SAME:      {columns = 4 : i32, rows = 4 : i32} :
// CHECK-SAME:      vector<16xf32> into vector<16xf32>
// CHECK:       return %[[T]] : vector<16xf32>

// -----

func @flat_transpose_index(%arg0: vector<16xindex>) -> vector<16xindex> {
  %0 = vector.flat_transpose %arg0 { rows = 4: i32, columns = 4: i32 }
     : vector<16xindex> -> vector<16xindex>
  return %0 : vector<16xindex>
}
// CHECK-LABEL: func @flat_transpose_index
// CHECK-SAME:  %[[A:.*]]: vector<16xindex>
// CHECK:       %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<16xindex> to vector<16xi64>
// CHECK:       %[[T1:.*]] = llvm.intr.matrix.transpose %[[T0]]
// CHECK-SAME:      {columns = 4 : i32, rows = 4 : i32} :
// CHECK-SAME:      vector<16xi64> into vector<16xi64>
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<16xi64> to vector<16xindex>
// CHECK:       return %[[T2]] : vector<16xindex>

// -----

func @vector_load_op(%memref : memref<200x100xf32>, %i : index, %j : index) -> vector<8xf32> {
  %0 = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func @vector_load_op
// CHECK: %[[c100:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK: %[[mul:.*]] = llvm.mul %{{.*}}, %[[c100]]  : i64
// CHECK: %[[add:.*]] = llvm.add %[[mul]], %{{.*}}  : i64
// CHECK: %[[gep:.*]] = llvm.getelementptr %{{.*}}[%[[add]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[bcast:.*]] = llvm.bitcast %[[gep]] : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
// CHECK: llvm.load %[[bcast]] {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>

// -----

func @vector_load_op_index(%memref : memref<200x100xindex>, %i : index, %j : index) -> vector<8xindex> {
  %0 = vector.load %memref[%i, %j] : memref<200x100xindex>, vector<8xindex>
  return %0 : vector<8xindex>
}
// CHECK-LABEL: func @vector_load_op_index
// CHECK: %[[T0:.*]] = llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<vector<8xi64>>
// CHECK: %[[T1:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<8xi64> to vector<8xindex>
// CHECK: return %[[T1]] : vector<8xindex>

// -----

func @vector_store_op(%memref : memref<200x100xf32>, %i : index, %j : index) {
  %val = arith.constant dense<11.0> : vector<4xf32>
  vector.store %val, %memref[%i, %j] : memref<200x100xf32>, vector<4xf32>
  return
}

// CHECK-LABEL: func @vector_store_op
// CHECK: %[[c100:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK: %[[mul:.*]] = llvm.mul %{{.*}}, %[[c100]]  : i64
// CHECK: %[[add:.*]] = llvm.add %[[mul]], %{{.*}}  : i64
// CHECK: %[[gep:.*]] = llvm.getelementptr %{{.*}}[%[[add]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[bcast:.*]] = llvm.bitcast %[[gep]] : !llvm.ptr<f32> to !llvm.ptr<vector<4xf32>>
// CHECK: llvm.store %{{.*}}, %[[bcast]] {alignment = 4 : i64} : !llvm.ptr<vector<4xf32>>

// -----

func @vector_store_op_index(%memref : memref<200x100xindex>, %i : index, %j : index) {
  %val = arith.constant dense<11> : vector<4xindex>
  vector.store %val, %memref[%i, %j] : memref<200x100xindex>, vector<4xindex>
  return
}
// CHECK-LABEL: func @vector_store_op_index
// CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} : !llvm.ptr<vector<4xi64>>

// -----

func @masked_load_op(%arg0: memref<?xf32>, %arg1: vector<16xi1>, %arg2: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0: index
  %0 = vector.maskedload %arg0[%c0], %arg1, %arg2 : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @masked_load_op
// CHECK: %[[CO:.*]] = arith.constant 0 : index
// CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[CO]] : index to i64
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%[[C]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[B:.*]] = llvm.bitcast %[[P]] : !llvm.ptr<f32> to !llvm.ptr<vector<16xf32>>
// CHECK: %[[L:.*]] = llvm.intr.masked.load %[[B]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.ptr<vector<16xf32>>, vector<16xi1>, vector<16xf32>) -> vector<16xf32>
// CHECK: return %[[L]] : vector<16xf32>

// -----

func @masked_load_op_index(%arg0: memref<?xindex>, %arg1: vector<16xi1>, %arg2: vector<16xindex>) -> vector<16xindex> {
  %c0 = arith.constant 0: index
  %0 = vector.maskedload %arg0[%c0], %arg1, %arg2 : memref<?xindex>, vector<16xi1>, vector<16xindex> into vector<16xindex>
  return %0 : vector<16xindex>
}
// CHECK-LABEL: func @masked_load_op_index
// CHECK: %{{.*}} = llvm.intr.masked.load %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : (!llvm.ptr<vector<16xi64>>, vector<16xi1>, vector<16xi64>) -> vector<16xi64>

// -----

func @masked_store_op(%arg0: memref<?xf32>, %arg1: vector<16xi1>, %arg2: vector<16xf32>) {
  %c0 = arith.constant 0: index
  vector.maskedstore %arg0[%c0], %arg1, %arg2 : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL: func @masked_store_op
// CHECK: %[[CO:.*]] = arith.constant 0 : index
// CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[CO]] : index to i64
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%[[C]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[B:.*]] = llvm.bitcast %[[P]] : !llvm.ptr<f32> to !llvm.ptr<vector<16xf32>>
// CHECK: llvm.intr.masked.store %{{.*}}, %[[B]], %{{.*}} {alignment = 4 : i32} : vector<16xf32>, vector<16xi1> into !llvm.ptr<vector<16xf32>>

// -----

func @masked_store_op_index(%arg0: memref<?xindex>, %arg1: vector<16xi1>, %arg2: vector<16xindex>) {
  %c0 = arith.constant 0: index
  vector.maskedstore %arg0[%c0], %arg1, %arg2 : memref<?xindex>, vector<16xi1>, vector<16xindex>
  return
}
// CHECK-LABEL: func @masked_store_op_index
// CHECK: llvm.intr.masked.store %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : vector<16xi64>, vector<16xi1> into !llvm.ptr<vector<16xi64>>

// -----

func @gather_op(%arg0: memref<?xf32>, %arg1: vector<3xi32>, %arg2: vector<3xi1>, %arg3: vector<3xf32>) -> vector<3xf32> {
  %0 = arith.constant 0: index
  %1 = vector.gather %arg0[%0][%arg1], %arg2, %arg3 : memref<?xf32>, vector<3xi32>, vector<3xi1>, vector<3xf32> into vector<3xf32>
  return %1 : vector<3xf32>
}

// CHECK-LABEL: func @gather_op
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<f32>, vector<3xi32>) -> !llvm.vec<3 x ptr<f32>>
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.vec<3 x ptr<f32>>, vector<3xi1>, vector<3xf32>) -> vector<3xf32>
// CHECK: return %[[G]] : vector<3xf32>

// -----

func @gather_op_index(%arg0: memref<?xindex>, %arg1: vector<3xindex>, %arg2: vector<3xi1>, %arg3: vector<3xindex>) -> vector<3xindex> {
  %0 = arith.constant 0: index
  %1 = vector.gather %arg0[%0][%arg1], %arg2, %arg3 : memref<?xindex>, vector<3xindex>, vector<3xi1>, vector<3xindex> into vector<3xindex>
  return %1 : vector<3xindex>
}

// CHECK-LABEL: func @gather_op_index
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<i64>, vector<3xi64>) -> !llvm.vec<3 x ptr<i64>>
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : (!llvm.vec<3 x ptr<i64>>, vector<3xi1>, vector<3xi64>) -> vector<3xi64>
// CHECK: %{{.*}} = builtin.unrealized_conversion_cast %[[G]] : vector<3xi64> to vector<3xindex>

// -----

func @gather_2d_op(%arg0: memref<4x4xf32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: vector<4xf32>) -> vector<4xf32> {
  %0 = arith.constant 3 : index
  %1 = vector.gather %arg0[%0, %0][%arg1], %arg2, %arg3 : memref<4x4xf32>, vector<4xi32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %1 : vector<4xf32>
}

// CHECK-LABEL: func @gather_2d_op
// CHECK: %[[B:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[P:.*]] = llvm.getelementptr %[[B]][%{{.*}}] : (!llvm.ptr<f32>, vector<4xi32>) -> !llvm.vec<4 x ptr<f32>>
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.vec<4 x ptr<f32>>, vector<4xi1>, vector<4xf32>) -> vector<4xf32>
// CHECK: return %[[G]] : vector<4xf32>

// -----

func @scatter_op(%arg0: memref<?xf32>, %arg1: vector<3xi32>, %arg2: vector<3xi1>, %arg3: vector<3xf32>) {
  %0 = arith.constant 0: index
  vector.scatter %arg0[%0][%arg1], %arg2, %arg3 : memref<?xf32>, vector<3xi32>, vector<3xi1>, vector<3xf32>
  return
}

// CHECK-LABEL: func @scatter_op
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<f32>, vector<3xi32>) -> !llvm.vec<3 x ptr<f32>>
// CHECK: llvm.intr.masked.scatter %{{.*}}, %[[P]], %{{.*}} {alignment = 4 : i32} : vector<3xf32>, vector<3xi1> into !llvm.vec<3 x ptr<f32>>

// -----

func @scatter_op_index(%arg0: memref<?xindex>, %arg1: vector<3xindex>, %arg2: vector<3xi1>, %arg3: vector<3xindex>) {
  %0 = arith.constant 0: index
  vector.scatter %arg0[%0][%arg1], %arg2, %arg3 : memref<?xindex>, vector<3xindex>, vector<3xi1>, vector<3xindex>
  return
}

// CHECK-LABEL: func @scatter_op_index
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<i64>, vector<3xi64>) -> !llvm.vec<3 x ptr<i64>>
// CHECK: llvm.intr.masked.scatter %{{.*}}, %[[P]], %{{.*}} {alignment = 8 : i32} : vector<3xi64>, vector<3xi1> into !llvm.vec<3 x ptr<i64>>

// -----

func @scatter_2d_op(%arg0: memref<4x4xf32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: vector<4xf32>) {
  %0 = arith.constant 3 : index
  vector.scatter %arg0[%0, %0][%arg1], %arg2, %arg3 : memref<4x4xf32>, vector<4xi32>, vector<4xi1>, vector<4xf32>
  return
}

// CHECK-LABEL: func @scatter_2d_op
// CHECK: %[[B:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[P:.*]] = llvm.getelementptr %[[B]][%{{.*}}] : (!llvm.ptr<f32>, vector<4xi32>) -> !llvm.vec<4 x ptr<f32>>
// CHECK: llvm.intr.masked.scatter %{{.*}}, %[[P]], %{{.*}} {alignment = 4 : i32} : vector<4xf32>, vector<4xi1> into !llvm.vec<4 x ptr<f32>>

// -----

func @expand_load_op(%arg0: memref<?xf32>, %arg1: vector<11xi1>, %arg2: vector<11xf32>) -> vector<11xf32> {
  %c0 = arith.constant 0: index
  %0 = vector.expandload %arg0[%c0], %arg1, %arg2 : memref<?xf32>, vector<11xi1>, vector<11xf32> into vector<11xf32>
  return %0 : vector<11xf32>
}

// CHECK-LABEL: func @expand_load_op
// CHECK: %[[CO:.*]] = arith.constant 0 : index
// CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[CO]] : index to i64
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%[[C]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[E:.*]] = "llvm.intr.masked.expandload"(%[[P]], %{{.*}}, %{{.*}}) : (!llvm.ptr<f32>, vector<11xi1>, vector<11xf32>) -> vector<11xf32>
// CHECK: return %[[E]] : vector<11xf32>

// -----

func @expand_load_op_index(%arg0: memref<?xindex>, %arg1: vector<11xi1>, %arg2: vector<11xindex>) -> vector<11xindex> {
  %c0 = arith.constant 0: index
  %0 = vector.expandload %arg0[%c0], %arg1, %arg2 : memref<?xindex>, vector<11xi1>, vector<11xindex> into vector<11xindex>
  return %0 : vector<11xindex>
}
// CHECK-LABEL: func @expand_load_op_index
// CHECK: %{{.*}} = "llvm.intr.masked.expandload"(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i64>, vector<11xi1>, vector<11xi64>) -> vector<11xi64>

// -----

func @compress_store_op(%arg0: memref<?xf32>, %arg1: vector<11xi1>, %arg2: vector<11xf32>) {
  %c0 = arith.constant 0: index
  vector.compressstore %arg0[%c0], %arg1, %arg2 : memref<?xf32>, vector<11xi1>, vector<11xf32>
  return
}

// CHECK-LABEL: func @compress_store_op
// CHECK: %[[CO:.*]] = arith.constant 0 : index
// CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[CO]] : index to i64
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%[[C]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: "llvm.intr.masked.compressstore"(%{{.*}}, %[[P]], %{{.*}}) : (vector<11xf32>, !llvm.ptr<f32>, vector<11xi1>) -> ()

// -----

func @compress_store_op_index(%arg0: memref<?xindex>, %arg1: vector<11xi1>, %arg2: vector<11xindex>) {
  %c0 = arith.constant 0: index
  vector.compressstore %arg0[%c0], %arg1, %arg2 : memref<?xindex>, vector<11xi1>, vector<11xindex>
  return
}
// CHECK-LABEL: func @compress_store_op_index
// CHECK: "llvm.intr.masked.compressstore"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<11xi64>, !llvm.ptr<i64>, vector<11xi1>) -> ()

// -----

// CHECK-LABEL: @splat_0d
// CHECK-SAME: %[[ARG:.*]]: f32
func @splat_0d(%a: f32) -> vector<f32> {
  %v = vector.splat %a : vector<f32>
  return %v : vector<f32>
}
// CHECK-NEXT: %[[UNDEF:[0-9]+]] = llvm.mlir.undef : vector<1xf32>
// CHECK-NEXT: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[V:[0-9]+]] = llvm.insertelement %[[ARG]], %[[UNDEF]][%[[ZERO]] : i32] : vector<1xf32>
// CHECK-NEXT: %[[VCAST:[0-9]+]] = builtin.unrealized_conversion_cast %[[V]] : vector<1xf32> to vector<f32>
// CHECK-NEXT: return %[[VCAST]] : vector<f32>

// -----

// CHECK-LABEL: @splat
// CHECK-SAME: %[[A:arg[0-9]+]]: vector<4xf32>
// CHECK-SAME: %[[ELT:arg[0-9]+]]: f32
func @splat(%a: vector<4xf32>, %b: f32) -> vector<4xf32> {
  %vb = vector.splat %b : vector<4xf32>
  %r = arith.mulf %a, %vb : vector<4xf32>
  return %r : vector<4xf32>
}
// CHECK-NEXT: %[[UNDEF:[0-9]+]] = llvm.mlir.undef : vector<4xf32>
// CHECK-NEXT: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[V:[0-9]+]] = llvm.insertelement %[[ELT]], %[[UNDEF]][%[[ZERO]] : i32] : vector<4xf32>
// CHECK-NEXT: %[[SPLAT:[0-9]+]] = llvm.shufflevector %[[V]], %[[UNDEF]] [0 : i32, 0 : i32, 0 : i32, 0 : i32]
// CHECK-NEXT: %[[SCALE:[0-9]+]] = arith.mulf %[[A]], %[[SPLAT]] : vector<4xf32>
// CHECK-NEXT: return %[[SCALE]] : vector<4xf32>
