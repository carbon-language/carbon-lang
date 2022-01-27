// RUN: toyc-ch7 %s -emit=mlir-affine 2>&1 | FileCheck %s
// RUN: toyc-ch7 %s -emit=mlir-affine -opt 2>&1 | FileCheck %s --check-prefix=OPT

func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}

// CHECK-LABEL: func @main()
// CHECK-DAG:     [[VAL_0:%.*]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:     [[VAL_1:%.*]] = arith.constant 2.000000e+00 : f64
// CHECK-DAG:     [[VAL_2:%.*]] = arith.constant 3.000000e+00 : f64
// CHECK-DAG:     [[VAL_3:%.*]] = arith.constant 4.000000e+00 : f64
// CHECK-DAG:     [[VAL_4:%.*]] = arith.constant 5.000000e+00 : f64
// CHECK-DAG:     [[VAL_5:%.*]] = arith.constant 6.000000e+00 : f64
// CHECK:         [[VAL_6:%.*]] = memref.alloc() : memref<3x2xf64>
// CHECK:         [[VAL_7:%.*]] = memref.alloc() : memref<3x2xf64>
// CHECK:         [[VAL_8:%.*]] = memref.alloc() : memref<2x3xf64>
// CHECK:         affine.store [[VAL_0]], [[VAL_8]][0, 0] : memref<2x3xf64>
// CHECK:         affine.store [[VAL_1]], [[VAL_8]][0, 1] : memref<2x3xf64>
// CHECK:         affine.store [[VAL_2]], [[VAL_8]][0, 2] : memref<2x3xf64>
// CHECK:         affine.store [[VAL_3]], [[VAL_8]][1, 0] : memref<2x3xf64>
// CHECK:         affine.store [[VAL_4]], [[VAL_8]][1, 1] : memref<2x3xf64>
// CHECK:         affine.store [[VAL_5]], [[VAL_8]][1, 2] : memref<2x3xf64>
// CHECK:         affine.for [[VAL_9:%.*]] = 0 to 3 {
// CHECK:           affine.for [[VAL_10:%.*]] = 0 to 2 {
// CHECK:             [[VAL_11:%.*]] = affine.load [[VAL_8]]{{\[}}[[VAL_10]], [[VAL_9]]] : memref<2x3xf64>
// CHECK:             affine.store [[VAL_11]], [[VAL_7]]{{\[}}[[VAL_9]], [[VAL_10]]] : memref<3x2xf64>
// CHECK:         affine.for [[VAL_12:%.*]] = 0 to 3 {
// CHECK:           affine.for [[VAL_13:%.*]] = 0 to 2 {
// CHECK:             [[VAL_14:%.*]] = affine.load [[VAL_7]]{{\[}}[[VAL_12]], [[VAL_13]]] : memref<3x2xf64>
// CHECK:             [[VAL_15:%.*]] = affine.load [[VAL_7]]{{\[}}[[VAL_12]], [[VAL_13]]] : memref<3x2xf64>
// CHECK:             [[VAL_16:%.*]] = arith.mulf [[VAL_14]], [[VAL_15]] : f64
// CHECK:             affine.store [[VAL_16]], [[VAL_6]]{{\[}}[[VAL_12]], [[VAL_13]]] : memref<3x2xf64>
// CHECK:         toy.print [[VAL_6]] : memref<3x2xf64>
// CHECK:         memref.dealloc [[VAL_8]] : memref<2x3xf64>
// CHECK:         memref.dealloc [[VAL_7]] : memref<3x2xf64>
// CHECK:         memref.dealloc [[VAL_6]] : memref<3x2xf64>

// OPT-LABEL: func @main()
// OPT-DAG:     [[VAL_0:%.*]] = arith.constant 1.000000e+00 : f64
// OPT-DAG:     [[VAL_1:%.*]] = arith.constant 2.000000e+00 : f64
// OPT-DAG:     [[VAL_2:%.*]] = arith.constant 3.000000e+00 : f64
// OPT-DAG:     [[VAL_3:%.*]] = arith.constant 4.000000e+00 : f64
// OPT-DAG:     [[VAL_4:%.*]] = arith.constant 5.000000e+00 : f64
// OPT-DAG:     [[VAL_5:%.*]] = arith.constant 6.000000e+00 : f64
// OPT:         [[VAL_6:%.*]] = memref.alloc() : memref<3x2xf64>
// OPT:         [[VAL_7:%.*]] = memref.alloc() : memref<2x3xf64>
// OPT:         affine.store [[VAL_0]], [[VAL_7]][0, 0] : memref<2x3xf64>
// OPT:         affine.store [[VAL_1]], [[VAL_7]][0, 1] : memref<2x3xf64>
// OPT:         affine.store [[VAL_2]], [[VAL_7]][0, 2] : memref<2x3xf64>
// OPT:         affine.store [[VAL_3]], [[VAL_7]][1, 0] : memref<2x3xf64>
// OPT:         affine.store [[VAL_4]], [[VAL_7]][1, 1] : memref<2x3xf64>
// OPT:         affine.store [[VAL_5]], [[VAL_7]][1, 2] : memref<2x3xf64>
// OPT:         affine.for [[VAL_8:%.*]] = 0 to 3 {
// OPT:           affine.for [[VAL_9:%.*]] = 0 to 2 {
// OPT:             [[VAL_10:%.*]] = affine.load [[VAL_7]]{{\[}}[[VAL_9]], [[VAL_8]]] : memref<2x3xf64>
// OPT:             [[VAL_11:%.*]] = arith.mulf [[VAL_10]], [[VAL_10]] : f64
// OPT:             affine.store [[VAL_11]], [[VAL_6]]{{\[}}[[VAL_8]], [[VAL_9]]] : memref<3x2xf64>
// OPT:         toy.print [[VAL_6]] : memref<3x2xf64>
// OPT:         memref.dealloc [[VAL_7]] : memref<2x3xf64>
// OPT:         memref.dealloc [[VAL_6]] : memref<3x2xf64>
