// RUN: mlir-opt %s -test-vector-to-vector-conversion | FileCheck %s

// CHECK-LABEL: func @add4x2
//      CHECK: %[[V1:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[V2:.*]] = vector.strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[V3:.*]] = vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[V4:.*]] = vector.strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf32> to vector<2x2xf32>
// CHECK-NEXT: %[[V5:.*]] = addf %[[V1]], %[[V3]] : vector<2x2xf32>
// CHECK-NEXT: %[[V6:.*]] = addf %[[V2]], %[[V4]] : vector<2x2xf32>
// CHECK-NEXT: "__fake_join__"(%[[V5]], %[[V6]]) {__base_coord__ = [0], __unroll__ = [2, 1]} : (vector<2x2xf32>, vector<2x2xf32>) -> vector<4x2xf32>
func @add4x2(%0: vector<4x2xf32>) -> vector<4x2xf32> {
  %1 = addf %0, %0: vector<4x2xf32>
  return %1: vector<4x2xf32>
}

// CHECK-LABEL: func @add4x4
//      CHECK: vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: vector.strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT: addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NEXT: "__fake_join__"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {__base_coord__ = [0], __unroll__ = [2, 2]} : (vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>) -> vector<4x4xf32>
func @add4x4(%0: vector<4x4xf32>, %1: vector<4x4xf32>) -> vector<4x4xf32> {
  %2 = addf %0, %1: vector<4x4xf32>
  %3 = addf %1, %2: vector<4x4xf32>
  return %3: vector<4x4xf32>
}
