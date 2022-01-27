// RUN: mlir-opt --split-input-file --tosa-to-linalg-named %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: @matmul
func @matmul(%arg0: tensor<1x5x3xf32>, %arg1: tensor<1x3x6xf32>) -> (tensor<1x5x6xf32>) {
  // CHECK: [[C0:%.+]] = arith.constant 0
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 5, 6]
  // CHECK: [[FILLED:%.+]] = linalg.fill([[C0]], [[INIT]]) : f32, tensor<1x5x6xf32> -> tensor<1x5x6xf32>
  // CHECK: linalg.batch_matmul ins(%arg0, %arg1 : tensor<1x5x3xf32>, tensor<1x3x6xf32>) outs([[FILLED]] : tensor<1x5x6xf32>) -> tensor<1x5x6xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x5x3xf32>, tensor<1x3x6xf32>)  -> (tensor<1x5x6xf32>)
  return %0 : tensor<1x5x6xf32>
}

// -----


// CHECK-LABEL: @matmul_quantized
func @matmul_quantized(%arg0: tensor<1x5x3xi8>, %arg1: tensor<1x3x6xi8>) -> (tensor<1x5x6xi32>) {
  // CHECK: [[C0:%.+]] = arith.constant 0
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 5, 6]
  // CHECK: [[FILLED:%.+]] = linalg.fill([[C0]], [[INIT]]) : i32, tensor<1x5x6xi32> -> tensor<1x5x6xi32>
  // CHECK: [[ONE:%.+]] = arith.constant 1
  // CHECK: [[TWO:%.+]] = arith.constant 2
  // CHECK: linalg.quantized_batch_matmul ins(%arg0, %arg1, [[ONE]], [[TWO]] : tensor<1x5x3xi8>, tensor<1x3x6xi8>, i32, i32) outs([[FILLED]] : tensor<1x5x6xi32>) -> tensor<1x5x6xi32>
  %0 = "tosa.matmul"(%arg0, %arg1) {quantization_info = {a_zp = 1 : i32, b_zp = 2 : i32}} : (tensor<1x5x3xi8>, tensor<1x3x6xi8>) -> (tensor<1x5x6xi32>)
  return %0 : tensor<1x5x6xi32>
}

// -----

// CHECK-LABEL: @matmul_dyn_batch
func @matmul_dyn_batch(%arg0: tensor<?x5x3xf32>, %arg1: tensor<?x3x6xf32>) -> (tensor<?x5x6xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[DIM:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[C0_0:.+]] = arith.constant 0
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [%[[DIM]], 5, 6]
  // CHECK: %[[FILLED:.+]] = linalg.fill(%[[C0_0]], %[[INIT]]) : f32, tensor<?x5x6xf32> -> tensor<?x5x6xf32>
  // CHECK: linalg.batch_matmul ins(%arg0, %arg1 : tensor<?x5x3xf32>, tensor<?x3x6xf32>) outs(%[[FILLED]] : tensor<?x5x6xf32>) -> tensor<?x5x6xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<?x5x3xf32>, tensor<?x3x6xf32>)  -> (tensor<?x5x6xf32>)
  return %0 : tensor<?x5x6xf32>
}

// -----

// CHECK-LABEL: @matmul_dyn_independent_dim
func @matmul_dyn_independent_dim(%arg0: tensor<1x5x3xf32>, %arg1: tensor<1x3x?xf32>) -> (tensor<1x5x?xf32>) {
  // CHECK: %[[C2:.+]] = arith.constant 2
  // CHECK: %[[DIM:.+]] = tensor.dim %arg1, %[[C2]]
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [1, 5, %[[DIM]]]
  // CHECK: %[[FILLED:.+]] = linalg.fill(%[[C0]], %[[INIT]]) : f32, tensor<1x5x?xf32> -> tensor<1x5x?xf32>
  // CHECK: linalg.batch_matmul ins(%arg0, %arg1 : tensor<1x5x3xf32>, tensor<1x3x?xf32>) outs(%[[FILLED]] : tensor<1x5x?xf32>) -> tensor<1x5x?xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x5x3xf32>, tensor<1x3x?xf32>)  -> (tensor<1x5x?xf32>)
  return %0 : tensor<1x5x?xf32>
}

// -----

// CHECK-LABEL: @matmul_dyn_independent_dim
func @matmul_dyn_independent_dim(%arg0: tensor<1x5x?xf32>, %arg1: tensor<1x?x6xf32>) -> (tensor<1x5x6xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [1, 5, 6]
  // CHECK: %[[FILLED:.+]] = linalg.fill(%[[C0]], %[[INIT]]) : f32, tensor<1x5x6xf32> -> tensor<1x5x6xf32>
  // CHECK: linalg.batch_matmul ins(%arg0, %arg1 : tensor<1x5x?xf32>, tensor<1x?x6xf32>) outs(%[[FILLED]] : tensor<1x5x6xf32>) -> tensor<1x5x6xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x5x?xf32>, tensor<1x?x6xf32>)  -> (tensor<1x5x6xf32>)
  return %0 : tensor<1x5x6xf32>
}

// -----

// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @fully_connected
func @fully_connected(%arg0: tensor<5x3xf32>, %arg1: tensor<6x3xf32>, %arg2: tensor<6xf32>) -> (tensor<5x6xf32>) {
  // CHECK: [[INITT:%.+]] = linalg.init_tensor [5, 6]
  // CHECK: [[ZERO:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[ZERO]], [[INITT]])
  // CHECK: [[PERM:%.+]] = arith.constant dense<[1, 0]>
  // CHECK: [[TRANSPOSE:%.+]] = "tosa.transpose"(%arg1, [[PERM]])
  // CHECK: [[INITB:%.+]] = linalg.init_tensor [5, 6]
  // CHECK: [[MATMUL:%.+]] = linalg.matmul ins(%arg0, [[TRANSPOSE]] : tensor<5x3xf32>, tensor<3x6xf32>) outs([[FILL]] : tensor<5x6xf32>) -> tensor<5x6xf32>
  // CHECK: [[ADDED:%.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%arg2, [[MATMUL]] : tensor<6xf32>, tensor<5x6xf32>) outs([[INITB]] : tensor<5x6xf32>) {
  // CHECK: ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
  // CHECK:   [[ADD:%.+]] = arith.addf %arg3, %arg4 : f32
  // CHECK:   linalg.yield [[ADD]] : f32

  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<5x3xf32>, tensor<6x3xf32>, tensor<6xf32>)  -> (tensor<5x6xf32>)
  return %0 : tensor<5x6xf32>
}

// -----

// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @quantized_fully_connected
func @quantized_fully_connected(%arg0: tensor<5x3xi8>, %arg1: tensor<6x3xi8>, %arg2: tensor<6xi32>) -> (tensor<5x6xi32>) {
  // CHECK: [[INITT:%.+]] = linalg.init_tensor [5, 6]
  // CHECK: [[ZERO:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[ZERO]], [[INITT]])
  // CHECK: [[PERM:%.+]] = arith.constant dense<[1, 0]>
  // CHECK: [[TRANSPOSE:%.+]] = "tosa.transpose"(%arg1, [[PERM]])
  // CHECK: [[INITB:%.+]] = linalg.init_tensor [5, 6]
  // CHECK: [[ONE:%.+]] = arith.constant 1
  // CHECK: [[TWO:%.+]] = arith.constant 2
  // CHECK: [[MATMUL:%.+]] = linalg.quantized_matmul ins(%arg0, [[TRANSPOSE]], [[ONE]], [[TWO]] : tensor<5x3xi8>, tensor<3x6xi8>, i32, i32) outs([[FILL]] : tensor<5x6xi32>) -> tensor<5x6xi32>
  // CHECK: [[ADDED:%.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%arg2, [[MATMUL]] : tensor<6xi32>, tensor<5x6xi32>) outs([[INITB]]
  // CHECK: ^bb0([[IN1:%.+]]: i32, [[IN2:%.+]]: i32, [[UNUSED:%.+]]: i32):
  // CHECK:   [[ADD:%.+]] = arith.addi
  // CHECK:   linalg.yield [[ADD]] : i32
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) {quantization_info = {input_zp = 1:i32, weight_zp = 2:i32}} : (tensor<5x3xi8>, tensor<6x3xi8>, tensor<6xi32>)  -> (tensor<5x6xi32>)
  return %0 : tensor<5x6xi32>
}

// -----

// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @fully_connected_dyn
func @fully_connected_dyn(%arg0: tensor<?x3xf32>, %arg1: tensor<6x3xf32>, %arg2: tensor<6xf32>) -> (tensor<?x6xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[DIM:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[INITT:.+]] = linalg.init_tensor [%[[DIM]], 6]
  // CHECK: %[[ZERO:.+]] = arith.constant 0
  // CHECK: %[[FILL:.+]] = linalg.fill(%[[ZERO]], %[[INITT]])
  // CHECK: %[[PERM:.+]] = arith.constant dense<[1, 0]>
  // CHECK: %[[TRANSPOSE:.+]] = "tosa.transpose"(%arg1, %[[PERM]])
  // CHECK: %[[INITB:.+]] = linalg.init_tensor [%[[DIM]], 6]
  // CHECK: %[[MATMUL:.+]] = linalg.matmul ins(%arg0, %[[TRANSPOSE]] : tensor<?x3xf32>, tensor<3x6xf32>) outs(%[[FILL]] : tensor<?x6xf32>) -> tensor<?x6xf32>
  // CHECK: %[[ADDED:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%arg2, %[[MATMUL]] : tensor<6xf32>, tensor<?x6xf32>) outs(%[[INITB]] : tensor<?x6xf32>) {
  // CHECK: ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
  // CHECK:   %[[ADD:.+]] = arith.addf %arg3, %arg4 : f32
  // CHECK:   linalg.yield %[[ADD]] : f32

  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<?x3xf32>, tensor<6x3xf32>, tensor<6xf32>)  -> (tensor<?x6xf32>)
  return %0 : tensor<?x6xf32>
}

// -----

// CHECK-LABEL: @max_pool
func @max_pool(%arg0: tensor<1x6x34x62xf32>) -> () {
  // CHECK-DAG: [[CONST:%.+]] = arith.constant -3.40282347E+38
  // CHECK-DAG: [[INIT:%.+]] = linalg.init_tensor [1, 4, 32, 62]
  // CHECK-DAG: [[FILL:%.+]] = linalg.fill([[CONST]], [[INIT]])
  // CHECK-DAG: [[KERNEL:%.+]] = linalg.init_tensor [3, 3]
  // CHECK: linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, [[KERNEL]] : tensor<1x6x34x62xf32>, tensor<3x3xf32>) outs([[FILL]] : tensor<1x4x32x62xf32>)
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 0], kernel = [3, 3], stride = [1, 1]} : (tensor<1x6x34x62xf32>)  -> (tensor<1x4x32x62xf32>)
  return
}

// CHECK-LABEL: @max_pool_padded
func @max_pool_padded(%arg0: tensor<1x6x34x62xf32>) -> () {
  // CHECK-DAG: [[CONST:%.+]] = arith.constant -3.40282347E+38 : f32
  // CHECK-DAG: [[PAD:%.+]] = tensor.pad %arg0 low[0, 0, 0, 0] high[0, 0, 1, 0]
  // CHECK-DAG:   tensor.yield [[CONST]]
  // CHECK-DAG: [[INITVAL:%.+]] = arith.constant -3.40282347E+38 : f32
  // CHECK-DAG: [[INIT:%.+]] = linalg.init_tensor [1, 4, 33, 62]
  // CHECK-DAG: [[FILL:%.+]] = linalg.fill([[INITVAL]], [[INIT]])
  // CHECK-DAG: [[KERNEL:%.+]] = linalg.init_tensor [3, 3]
  // CHECK: linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins([[PAD]], [[KERNEL]] : tensor<1x6x35x62xf32>, tensor<3x3xf32>) outs([[FILL]] : tensor<1x4x33x62xf32>)
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 1], kernel = [3, 3], stride = [1, 1]} : (tensor<1x6x34x62xf32>)  -> (tensor<1x4x33x62xf32>)
  return
}

// CHECK-LABEL: @max_pool_dyn
func @max_pool_dyn(%arg0: tensor<?x6x34x62xf32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[CONST:.+]] = arith.constant -3.40282347E+38
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [%[[BATCH]], 4, 32, 62]
  // CHECK: %[[FILL:.+]] = linalg.fill(%[[CONST]], %[[INIT]])
  // CHECK: %[[KERNEL:.+]] = linalg.init_tensor [3, 3]
  // CHECK: linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %[[KERNEL]] : tensor<?x6x34x62xf32>, tensor<3x3xf32>) outs(%[[FILL]] : tensor<?x4x32x62xf32>)
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 0], kernel = [3, 3], stride = [1, 1]} : (tensor<?x6x34x62xf32>)  -> (tensor<?x4x32x62xf32>)
  return
}

// CHECK-LABEL: @max_pool_i8
func @max_pool_i8(%arg0: tensor<1x6x34x62xi8>) -> () {
  // CHECK: arith.constant -128
  // CHECK: linalg.pooling_nhwc_max
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 0], kernel = [3, 3], stride = [1, 1]} : (tensor<1x6x34x62xi8>)  -> (tensor<1x4x32x62xi8>)
  return
}

// CHECK-LABEL: @max_pool_i16
func @max_pool_i16(%arg0: tensor<1x6x34x62xi16>) -> () {
  // CHECK: arith.constant -32768
  // CHECK: linalg.pooling_nhwc_max
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 0], kernel = [3, 3], stride = [1, 1]} : (tensor<1x6x34x62xi16>)  -> (tensor<1x4x32x62xi16>)
  return
}

// CHECK-LABEL: @max_pool_i32
func @max_pool_i32(%arg0: tensor<1x6x34x62xi32>) -> () {
  // CHECK: arith.constant -2147483648
  // CHECK: linalg.pooling_nhwc_max
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 0], kernel = [3, 3], stride = [1, 1]} : (tensor<1x6x34x62xi32>)  -> (tensor<1x4x32x62xi32>)
  return
}
// -----

// CHECK-LABEL: @avg_pool
func @avg_pool(%arg0: tensor<1x6x34x62xf32>) -> (tensor<1x5x33x62xf32>) {
  // Initial piece computes the sum of the pooling region, with appropriate padding.
  // CHECK: [[CONST:%.+]] = arith.constant 0
  // CHECK: [[PAD:%.+]] = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]
  // CHECK: [[CONST:%.+]] = arith.constant 0
  // CHECK: [[POOLINIT:%.+]] = linalg.init_tensor [1, 5, 33, 62]
  // CHECK: [[FILL:%.+]] = linalg.fill([[CONST]], [[POOLINIT]])
  // CHECK: [[KERNEL:%.+]] = linalg.init_tensor [4, 4]
  // CHECK: [[POOL:%.+]] = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins([[PAD]], [[KERNEL]] : tensor<1x8x36x62xf32>, tensor<4x4xf32>) outs([[FILL]] : tensor<1x5x33x62xf32>)
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 5, 33, 62]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[POOL]] : tensor<1x5x33x62xf32>) outs([[INIT]] : tensor<1x5x33x62xf32>)
  // CHECK:   [[ZERO:%.0]] = arith.constant 0
  // CHECK:   [[ONE:%.+]] = arith.constant 1
  // CHECK:   [[HEIGHT:%.+]] = arith.constant 4
  // CHECK:   [[WIDTH:%.+]] = arith.constant 32
  // CHECK:   [[IDX1:%.+]] = linalg.index 1
  // CHECK:   [[IDX2:%.+]] = linalg.index 2

  // The large block below computes what portion of the kernel is within non-padded input.
  // CHECK:   [[NY:%.+]] = arith.subi [[HEIGHT]], [[IDX1]]
  // CHECK:   [[NX:%.+]] = arith.subi [[WIDTH]], [[IDX2]]
  // CHECK:   [[KH:%.+]] = arith.constant 4
  // CHECK:   [[PAD0:%.+]] = arith.constant 1
  // CHECK:   [[SUBP0:%.+]] = arith.subi [[IDX1]], [[PAD0]]
  // CHECK:   [[P0CMP:%.+]] = arith.cmpi slt, [[SUBP0]], [[ZERO]]
  // CHECK:   [[SELP0:%.+]] = select [[P0CMP]], [[SUBP0]], [[ZERO]]
  // CHECK:   [[ADDP0:%.+]] = arith.addi [[KH]], [[SELP0]]
  // CHECK:   [[PAD1:%.+]] = arith.constant 1
  // CHECK:   [[SUBP1:%.+]] = arith.subi [[NY]], [[PAD1]]
  // CHECK:   [[P1CMP:%.+]] = arith.cmpi slt, [[SUBP1]], [[ZERO]]
  // CHECK:   [[SELP1:%.+]] = select [[P1CMP]], [[SUBP1]], [[ZERO]]
  // CHECK:   [[ADDP1:%.+]] = arith.addi [[ADDP0]], [[SELP1]]
  // CHECK:   [[YCMP:%.+]] = arith.cmpi slt, [[ADDP1]], [[ONE]]
  // CHECK:   [[YSEL:%.+]] = select [[YCMP]], [[ONE]], [[ADDP1]]
  // CHECK:   [[KW:%.+]] = arith.constant 4 : index
  // CHECK:   [[PAD2:%.+]] = arith.constant 1 : index
  // CHECK:   [[SUBP2:%.+]] = arith.subi [[IDX2]], [[PAD2]]
  // CHECK:   [[P2CMP:%.+]] = arith.cmpi slt, [[SUBP2]], [[ZERO]]
  // CHECK:   [[SELP2:%.+]] = select [[P2CMP]], [[SUBP2]], [[ZERO]]
  // CHECK:   [[ADDP2:%.+]] = arith.addi [[KW]], [[SELP2]]
  // CHECK:   [[PAD3:%.+]] = arith.constant 1 : index
  // CHECK:   [[SUBP3:%.+]] = arith.subi [[NX]], [[PAD3]]
  // CHECK:   [[P3CMP:%.+]] = arith.cmpi slt, [[SUBP3]], [[ZERO]]
  // CHECK:   [[SELP3:%.+]] = select [[P3CMP]], [[SUBP3]], [[ZERO]]
  // CHECK:   [[ADDP3:%.+]] = arith.addi [[ADDP2]], [[SELP3]]
  // CHECK:   [[XCMP:%.+]] = arith.cmpi slt, [[ADDP3]], [[ONE]]
  // CHECK:   [[XSEL:%.+]] = select [[XCMP]], [[ONE]], [[ADDP3]]

  // Given the valid coverage of the pooling region, normalize the summation.
  // CHECK:   [[C:%.+]] = arith.muli [[YSEL]], [[XSEL]]
  // CHECK:   [[CI:%.+]] = arith.index_cast [[C]]
  // CHECK:   [[CF:%.+]] = arith.sitofp [[CI]]
  // CHECK:   [[RESULT:%.+]] = arith.divf %arg1, [[CF]]
  // CHECK:   linalg.yield [[RESULT]]
  %0 = "tosa.avg_pool2d"(%arg0) {pad = [1, 1, 1, 1], kernel = [4, 4], stride = [1, 1]} : (tensor<1x6x34x62xf32>)  -> (tensor<1x5x33x62xf32>)
  return %0 : tensor<1x5x33x62xf32>
}

// -----

// CHECK-LABEL: @avg_pool_dyn
func @avg_pool_dyn(%arg0: tensor<?x6x34x62xf32>) -> (tensor<?x5x33x62xf32>) {
  // The calculations remain the same as above, only testing for dyn behavior
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[PAD:.+]] = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]
  // CHECK: %[[POOLINIT:.+]] = linalg.init_tensor [%[[BATCH]], 5, 33, 62]
  // CHECK: %[[FILL:.+]] = linalg.fill
  // CHECK: %[[KERNEL:.+]] = linalg.init_tensor [4, 4]
  // CHECK: %[[POOL:.+]] = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%[[PAD]], %[[KERNEL]] : tensor<?x8x36x62xf32>, tensor<4x4xf32>) outs(%[[FILL]] : tensor<?x5x33x62xf32>)
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [%[[BATCH]], 5, 33, 62]
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[POOL]] : tensor<?x5x33x62xf32>) outs(%[[INIT]] : tensor<?x5x33x62xf32>)
  %0 = "tosa.avg_pool2d"(%arg0) {pad = [1, 1, 1, 1], kernel = [4, 4], stride = [1, 1]} : (tensor<?x6x34x62xf32>)  -> (tensor<?x5x33x62xf32>)
  return %0 : tensor<?x5x33x62xf32>
}

// -----

// CHECK-LABEL: @avg_pool_i8
func @avg_pool_i8(%arg0 : tensor<1x128x128x2xi8>) -> () {

  // CHECK: linalg.pooling_nhwc_sum
  // CHECK: linalg.generic

  // CHECK: %[[INZP:.+]] = arith.constant -128
  // CHECK: %[[INZP_OFF:.+]] = arith.muli %{{.+}}, %[[INZP]]
  // CHECK: %[[OFFSETED:.+]] = arith.subi %arg1, %[[INZP_OFF]]
  // CHECK: %[[NUMERATOR:.+]] = arith.constant 1073741825
  // CHECK: %[[MULTIPLIER:.+]] = arith.divui %[[NUMERATOR]], %{{.+}}
  // CHECK: %[[SHIFT:.+]] = arith.constant 30
  // CHECK: %[[SCALE:.+]] = "tosa.apply_scale"(%{{.+}}, %[[MULTIPLIER]], %[[SHIFT]]) {double_round = false}
  // CHECK: %[[OUTZP:.+]] = arith.constant -128
  // CHECK: %[[OUT:.+]] = arith.addi %[[SCALE]], %[[OUTZP]]
  // CHECK: %[[MIN:.+]] = arith.constant -128
  // CHECK: %[[MAX:.+]] = arith.constant 127
  // CHECK: %[[CMP_MIN:.+]] = arith.cmpi slt, %[[OUT]], %[[MIN]]
  // CHECK: %[[CLMP_MIN:.+]] = select %[[CMP_MIN]], %[[MIN]], %[[OUT]]
  // CHECK: %[[CMP_MAX:.+]] = arith.cmpi slt, %[[MAX]], %[[OUT]]
  // CHECK: %[[CLMP_MAX:.+]] = select %[[CMP_MAX]], %[[MAX]], %[[CLMP_MIN]]
  // CHECK: %[[TRUNC:.+]] = arith.trunci %[[CLMP_MAX]]
  // CHECK: linalg.yield %[[TRUNC]]
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = [4, 4], pad = [0, 0, 0, 0], quantization_info = {input_zp = -128 : i32, output_zp = -128 : i32}, stride = [4, 4]} : (tensor<1x128x128x2xi8>) -> tensor<1x32x32x2xi8>
  return
}

// -----

// CHECK-LABEL: @avg_pool_i16
func @avg_pool_i16(%arg0 : tensor<1x128x128x2xi16>) -> () {

  // CHECK: linalg.pooling_nhwc_sum
  // CHECK: linalg.generic

  // CHECK: %[[INZP:.+]] = arith.constant -128
  // CHECK: %[[INZP_OFF:.+]] = arith.muli %{{.+}}, %[[INZP]]
  // CHECK: %[[OFFSETED:.+]] = arith.subi %arg1, %[[INZP_OFF]]
  // CHECK: %[[NUMERATOR:.+]] = arith.constant 1073741825
  // CHECK: %[[MULTIPLIER:.+]] = arith.divui %[[NUMERATOR]], %{{.+}}
  // CHECK: %[[SHIFT:.+]] = arith.constant 30
  // CHECK: %[[SCALE:.+]] = "tosa.apply_scale"(%{{.+}}, %[[MULTIPLIER]], %[[SHIFT]]) {double_round = false}
  // CHECK: %[[OUTZP:.+]] = arith.constant -128
  // CHECK: %[[OUT:.+]] = arith.addi %[[SCALE]], %[[OUTZP]]
  // CHECK: %[[MIN:.+]] = arith.constant -32768
  // CHECK: %[[MAX:.+]] = arith.constant 32767
  // CHECK: %[[CMP_MIN:.+]] = arith.cmpi slt, %[[OUT]], %[[MIN]]
  // CHECK: %[[CLMP_MIN:.+]] = select %[[CMP_MIN]], %[[MIN]], %[[OUT]]
  // CHECK: %[[CMP_MAX:.+]] = arith.cmpi slt, %[[MAX]], %[[OUT]]
  // CHECK: %[[CLMP_MAX:.+]] = select %[[CMP_MAX]], %[[MAX]], %[[CLMP_MIN]]
  // CHECK: %[[TRUNC:.+]] = arith.trunci %[[CLMP_MAX]]
  // CHECK: linalg.yield %[[TRUNC]]
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = [4, 4], pad = [0, 0, 0, 0], quantization_info = {input_zp = -128 : i32, output_zp = -128 : i32}, stride = [4, 4]} : (tensor<1x128x128x2xi16>) -> tensor<1x32x32x2xi16>
  return
}

// -----

// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @conv2d_f32
func @conv2d_f32(%input: tensor<1x49x42x27xf32>, %weights: tensor<28x3x3x27xf32>, %bias: tensor<28xf32>) -> () {
  // CHECK: %[[PERM:.+]] = arith.constant dense<[1, 2, 3, 0]>
  // CHECK: %[[W:.+]] = "tosa.transpose"(%arg1, %[[PERM]])
  // CHECK: %[[M_IN:.+]] = linalg.init_tensor [1, 45, 40, 28]
  // CHECK: %[[CST:.+]] = arith.constant 0
  // CHECK: %[[FILL:.+]] = linalg.fill
  // CHECK: %[[B_IN:.+]] = linalg.init_tensor [1, 45, 40, 28]
  // CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<[2, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %[[W]] : tensor<1x49x42x27xf32>, tensor<3x3x27x28xf32>) outs(%[[FILL]] : tensor<1x45x40x28xf32>)
  // CHECK: %[[B:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %[[CONV]] : tensor<28xf32>, tensor<1x45x40x28xf32>) outs(%[[B_IN]] : tensor<1x45x40x28xf32>)
  // CHECK:   arith.addf
  // CHECK:   linalg.yield
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [2, 1]} : (tensor<1x49x42x27xf32>, tensor<28x3x3x27xf32>, tensor<28xf32>)  -> (tensor<1x45x40x28xf32>)
  return
}

// -----

// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @conv2d_dyn
func @conv2d_dyn(%input: tensor<?x49x42x27xf32>, %weights: tensor<28x3x3x27xf32>, %bias: tensor<28xf32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[PERM:.+]] = arith.constant dense<[1, 2, 3, 0]>
  // CHECK: %[[W:.+]] = "tosa.transpose"(%arg1, %[[PERM]])
  // CHECK: %[[M_IN:.+]] = linalg.init_tensor [%[[BATCH]], 45, 40, 28]
  // CHECK: %[[CST:.+]] = arith.constant 0
  // CHECK: %[[FILL:.+]] = linalg.fill
  // CHECK: %[[B_IN:.+]] = linalg.init_tensor [%[[BATCH]], 45, 40, 28]
  // CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<[2, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %[[W]] : tensor<?x49x42x27xf32>, tensor<3x3x27x28xf32>) outs(%[[FILL]] : tensor<?x45x40x28xf32>)
  // CHECK: %[[B:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %[[CONV]] : tensor<28xf32>, tensor<?x45x40x28xf32>) outs(%[[B_IN]] : tensor<?x45x40x28xf32>)
  // CHECK:   %[[ADD:.+]] = arith.addf
  // CHECK:   linalg.yield %[[ADD]] : f32
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [2, 1]} : (tensor<?x49x42x27xf32>, tensor<28x3x3x27xf32>, tensor<28xf32>)  -> (tensor<?x45x40x28xf32>)
  return
}

// -----

// CHECK-LABEL: @conv2d_padded_f32
func @conv2d_padded_f32(%input: tensor<1x47x40x28xf32>, %weights: tensor<28x3x3x28xf32>, %bias: tensor<28xf32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]
  // CHECK:   tensor.yield %[[C0]]
  // CHECK: linalg.conv_2d_nhwc_hwcf
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [1, 1, 1, 1], stride = [1, 1], dilation = [2, 1]} : (tensor<1x47x40x28xf32>, tensor<28x3x3x28xf32>, tensor<28xf32>)  -> (tensor<1x45x40x28xf32>)
  return
}

// -----

// CHECK-LABEL: @conv2d_quant
func @conv2d_quant(%arg0 : tensor<1x12x12x1xi8>, %arg1 : tensor<1024x3x3x1xi8>, %arg2 : tensor<1024xi32>) -> () {
  // CHECK:   %[[C22:.+]] = arith.constant -22
  // CHECK: tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]
  // CHECK:   tensor.yield %[[C22]]
  // CHECK: linalg.conv_2d_nhwc_hwcf_q
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [1, 1, 1, 1], quantization_info = {input_zp = -22 : i32, weight_zp = 42 : i32}, stride = [1, 1]} : (tensor<1x12x12x1xi8>, tensor<1024x3x3x1xi8>, tensor<1024xi32>) -> tensor<1x12x12x1024xi32>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @depthwise_conv
func @depthwise_conv(%arg0 : tensor<1x7x5x3xf32>, %arg1 : tensor<3x1x3x11xf32>, %arg2 : tensor<33xf32>) -> () {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 5, 5, 3, 11]
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[CST0]], [[INIT]])
  // CHECK: [[OUT:%.+]] = linalg.init_tensor [1, 5, 5, 33]
  // CHECK: [[DEPTH:%.+]] = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x7x5x3xf32>, tensor<3x1x3x11xf32>) outs([[FILL]] : tensor<1x5x5x3x11xf32>)
  // CHECK: [[COLLAPSED:%.+]] = "tosa.reshape"([[DEPTH]]) {new_shape = [1, 5, 5, 33]}
  // CHECK: [[BIAS:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, [[COLLAPSED]] : tensor<33xf32>, tensor<1x5x5x33xf32>) outs([[OUT]] : tensor<1x5x5x33xf32>) {
  // CHECK: ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  
  // CHECK:   [[ADD:%.+]] = arith.addf %arg3, %arg4 : f32
  // CHECK:   linalg.yield [[ADD]] : f32
  // CHECK: } -> tensor<1x5x5x33xf32>
  %2 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) { pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1] } : (tensor<1x7x5x3xf32>, tensor<3x1x3x11xf32>, tensor<33xf32>)  -> (tensor<1x5x5x33xf32>)
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @depthwise_conv_dyn
func @depthwise_conv_dyn(%arg0 : tensor<?x7x5x3xf32>, %arg1 : tensor<3x1x3x11xf32>, %arg2 : tensor<33xf32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [%[[BATCH]], 5, 5, 3, 11]
  // CHECK: %[[CST0:.+]] = arith.constant 0
  // CHECK: %[[FILL:.+]] = linalg.fill
  // CHECK: %[[OUT:.+]] = linalg.init_tensor [%[[BATCH]], 5, 5, 33]
  // CHECK: %[[DEPTH:.+]] = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<?x7x5x3xf32>, tensor<3x1x3x11xf32>) outs(%[[FILL]] : tensor<?x5x5x3x11xf32>)
  // CHECK: %[[COLLAPSED:.+]] = "tosa.reshape"(%[[DEPTH]]) {new_shape = [-1, 5, 5, 33]}
  // CHECK: %[[BIAS:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %[[COLLAPSED]] : tensor<33xf32>, tensor<?x5x5x33xf32>) outs(%[[OUT]] : tensor<?x5x5x33xf32>) {
  // CHECK: ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  
  // CHECK:   %[[ADD:.+]] = arith.addf %arg3, %arg4 : f32
  // CHECK:   linalg.yield %[[ADD]] : f32
  // CHECK: } -> tensor<?x5x5x33xf32>
  %2 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) { pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1] } : (tensor<?x7x5x3xf32>, tensor<3x1x3x11xf32>, tensor<33xf32>)  -> (tensor<?x5x5x33xf32>)
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @depthwise_conv_strides
func @depthwise_conv_strides(%arg0 : tensor<1x11x9x3xf32>, %arg1 : tensor<3x1x3x11xf32>, %arg2 : tensor<33xf32>) -> () {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 5, 5, 3, 11]
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[CST0]], [[INIT]])
  // CHECK: [[OUT:%.+]] = linalg.init_tensor [1, 5, 5, 33]
  // CHECK: [[DEPTH:%.+]] = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x11x9x3xf32>, tensor<3x1x3x11xf32>) outs([[FILL]] : tensor<1x5x5x3x11xf32>)
  // CHECK: [[COLLAPSED:%.+]] = "tosa.reshape"([[DEPTH]]) {new_shape = [1, 5, 5, 33]}
  // CHECK: [[BIAS:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, [[COLLAPSED]] : tensor<33xf32>, tensor<1x5x5x33xf32>) outs([[OUT]] : tensor<1x5x5x33xf32>) {
  // CHECK: ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  
  // CHECK:   [[ADD:%.+]] = arith.addf %arg3, %arg4 : f32
  // CHECK:   linalg.yield [[ADD]] : f32
  // CHECK: } -> tensor<1x5x5x33xf32>
  %2 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) { pad = [0, 0, 0, 0], stride = [2, 2], dilation = [1, 1] } : (tensor<1x11x9x3xf32>, tensor<3x1x3x11xf32>, tensor<33xf32>)  -> (tensor<1x5x5x33xf32>)
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @depthwise_conv_quant
func @depthwise_conv_quant(%arg0 : tensor<1x12x12x4xi8>, %arg1 : tensor<3x3x4x128xi8>, %arg2 : tensor<512xi32>) -> () {
  // CHECK: [[PADV:%.+]] = arith.constant -128
  // CHECK: [[PAD:%.+]] = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]
  // CHECK:   tensor.yield [[PADV]]

  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 12, 12, 4, 128]
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[CST0]], [[INIT]])
  // CHECK: [[OUT:%.+]] = linalg.init_tensor [1, 12, 12, 512]
  // CHECK: [[C128:%.+]] = arith.constant -128
  // CHECK: [[C42:%.+]] = arith.constant 42
  // CHECK: [[DEPTH:%.+]] = linalg.depthwise_conv_2d_nhwc_hwcm_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins([[PAD]], %arg1, [[C128]], [[C42]] : tensor<1x14x14x4xi8>, tensor<3x3x4x128xi8>, i32, i32) outs([[FILL]] : tensor<1x12x12x4x128xi32>)
  // CHECK: [[COLLAPSED:%.+]] = "tosa.reshape"([[DEPTH]]) {new_shape = [1, 12, 12, 512]}
  // CHECK: [[BIAS:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, [[COLLAPSED]] : tensor<512xi32>, tensor<1x12x12x512xi32>) outs([[OUT]] : tensor<1x12x12x512xi32>) {
  // CHECK: ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):  
  // CHECK:   [[ADD:%.+]] = arith.addi %arg3, %arg4 : i32
  // CHECK:   linalg.yield [[ADD]] : i32
  // CHECK: } -> tensor<1x12x12x512xi32>
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [1, 1, 1, 1], quantization_info = {input_zp = -128 : i32, weight_zp = 42 : i32}, stride = [1, 1], dilation = [1, 1] } : (tensor<1x12x12x4xi8>, tensor<3x3x4x128xi8>, tensor<512xi32>)  -> tensor<1x12x12x512xi32>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @depthwise_conv_quant_dilations
func @depthwise_conv_quant_dilations(%arg0 : tensor<1x14x14x4xi8>, %arg1 : tensor<3x3x4x128xi8>, %arg2 : tensor<512xi32>) -> () {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 10, 10, 4, 128]
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[CST0]], [[INIT]])
  // CHECK: [[OUT:%.+]] = linalg.init_tensor [1, 10, 10, 512]
  // CHECK: [[C128:%.+]] = arith.constant -128
  // CHECK: [[C42:%.+]] = arith.constant 42
  // CHECK: [[DEPTH:%.+]] = linalg.depthwise_conv_2d_nhwc_hwcm_q {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1, [[C128]], [[C42]] : tensor<1x14x14x4xi8>, tensor<3x3x4x128xi8>, i32, i32) outs([[FILL]] : tensor<1x10x10x4x128xi32>)
  // CHECK: [[COLLAPSED:%.+]] = "tosa.reshape"([[DEPTH]]) {new_shape = [1, 10, 10, 512]}
  // CHECK: [[BIAS:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, [[COLLAPSED]] : tensor<512xi32>, tensor<1x10x10x512xi32>) outs([[OUT]] : tensor<1x10x10x512xi32>) {
  // CHECK: ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):  
  // CHECK:   [[ADD:%.+]] = arith.addi %arg3, %arg4 : i32
  // CHECK:   linalg.yield [[ADD]] : i32
  // CHECK: } -> tensor<1x10x10x512xi32>
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [0, 0, 0, 0], quantization_info = {input_zp = -128 : i32, weight_zp = 42 : i32}, stride = [1, 1], dilation = [2, 2] } : (tensor<1x14x14x4xi8>, tensor<3x3x4x128xi8>, tensor<512xi32>)  -> tensor<1x10x10x512xi32>
  return
}
