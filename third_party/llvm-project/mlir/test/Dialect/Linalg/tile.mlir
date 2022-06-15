// RUN: mlir-opt %s -linalg-tile="tile-sizes=2" -mlir-disable-threading=true | FileCheck %s -check-prefix=TILE-2
// RUN: mlir-opt %s -linalg-tile="tile-sizes=0,2" -mlir-disable-threading=true | FileCheck %s -check-prefix=TILE-02
// RUN: mlir-opt %s -linalg-tile="tile-sizes=0,0,2" -mlir-disable-threading=true | FileCheck %s -check-prefix=TILE-002
// RUN: mlir-opt %s -linalg-tile="tile-sizes=2,3,4" -mlir-disable-threading=true | FileCheck %s -check-prefix=TILE-234

//   TILE-2-DAG: #[[$strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
//  TILE-02-DAG: #[[$strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// TILE-002-DAG: #[[$strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// TILE-234-DAG: #[[$strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

//   TILE-2-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
//  TILE-02-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// TILE-002-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// TILE-234-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>

//   TILE-2-DAG: #[[$bound_map:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
//  TILE-02-DAG: #[[$bound_map:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
// TILE-002-DAG: #[[$bound_map:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
// TILE-234-DAG: #[[$bound_map_2:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
// TILE-234-DAG: #[[$bound_map_3:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 3)>
// TILE-234-DAG: #[[$bound_map_4:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 4)>

//   TILE-2-DAG: #[[$stride_99_1_layout_map:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 99 + s0 + d1)>
//  TILE-02-DAG: #[[$stride_99_1_layout_map:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 99 + s0 + d1)>
// TILE-234-DAG: #[[$stride_99_1_layout_map:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 99 + s0 + d1)>

func.func @matmul(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %arg1: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %arg2: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.matmul
    ins(%arg0, %arg1: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                      memref<?x?xf32, offset: ?, strides: [?, 1]>)
   outs(%arg2: memref<?x?xf32, offset: ?, strides: [?, 1]>)
  return
}
// TILE-2-LABEL: func @matmul(
//       TILE-2-DAG: %[[C0:.*]] = arith.constant 0 : index
//       TILE-2-DAG: %[[C2:.*]] = arith.constant 2 : index
//       TILE-2: %[[M:.*]] = memref.dim %{{.*}}, %c0 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-2: scf.for %[[I:.*]] = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[szM:.*]] = affine.min #[[$bound_map]](%[[I]])[%[[M]]]
//       TILE-2:   %[[K:.*]] = memref.dim %{{.*}}, %c1 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-2:   %[[sAi:.*]] = memref.subview %{{.*}}[%[[I]], 0] [%[[szM]], %[[K]]] [1, 1] : memref<?x?xf32, #[[$strided2D]]> to memref<?x?xf32, #[[$strided2D]]>
//       TILE-2:   %[[szK:.*]] = affine.min #[[$bound_map]](%[[I]])[%[[M]]]
//       TILE-2:   %[[N:.*]] = memref.dim %{{.*}}, %c1 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-2:   %[[sCi:.*]] = memref.subview %{{.*}}[%[[I]], 0] [%[[szK]], %[[N]]] [1, 1] : memref<?x?xf32, #[[$strided2D]]> to memref<?x?xf32, #[[$strided2D]]>
//       TILE-2:   linalg.matmul ins(%[[sAi]]{{.*}} outs(%[[sCi]]

// TILE-02-LABEL: func @matmul(
//       TILE-02-DAG: %[[C0:.*]] = arith.constant 0 : index
//       TILE-02-DAG: %[[C2:.*]] = arith.constant 2 : index
//       TILE-02: %[[N:.*]] = memref.dim %arg1, %c1 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-02: scf.for %[[J:.*]] = %{{.*}} to %[[N]] step %{{.*}} {
//       TILE-02:   %[[K:.*]] = memref.dim %{{.*}}, %c0 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-02:   %[[szN:.*]] = affine.min #[[$bound_map]](%[[J]])[%[[N]]]
//       TILE-02:   %[[sBj:.*]] = memref.subview %{{.*}}[0, %[[J]]] [%[[K]], %[[szN]]] [1, 1] : memref<?x?xf32, #[[$strided2D]]> to memref<?x?xf32, #[[$strided2D]]>
//       TILE-02:   %[[M:.*]] = memref.dim %{{.*}}, %c0 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-02:   %[[szK:.*]] = affine.min #[[$bound_map]](%[[J]])[%[[N]]]
//       TILE-02:   %[[sCj:.*]] = memref.subview %{{.*}}[0, %[[J]]] [%[[M]], %[[szK]]] [1, 1] : memref<?x?xf32, #[[$strided2D]]> to memref<?x?xf32, #[[$strided2D]]>
//       TILE-02:   linalg.matmul ins(%{{.*}}, %[[sBj]]{{.*}} outs(%[[sCj]]

// TILE-002-LABEL: func @matmul(
//       TILE-002-DAG: %[[C0:.*]] = arith.constant 0 : index
//       TILE-002-DAG: %[[C2:.*]] = arith.constant 2 : index
//       TILE-002: %[[ubK:.*]] = memref.dim %{{.*}}, %c1 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-002: scf.for %[[K:.*]] = %{{.*}}{{.*}} to %[[ubK]] step %{{.*}} {
//       TILE-002:   %[[M:.*]] = memref.dim %{{.*}}, %c0 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-002:   %[[szK:.*]] = affine.min #[[$bound_map]](%[[K]])[%[[ubK]]]
//       TILE-002:   %[[sAj:.*]] = memref.subview %{{.*}}[0, %[[K]]] [%[[M]], %[[szK]]] [1, 1] : memref<?x?xf32, #[[$strided2D]]> to memref<?x?xf32, #[[$strided2D]]>
//       TILE-002:   %[[szK:.*]] = affine.min #[[$bound_map]](%[[K]])[%[[ubK]]]
//       TILE-002:   %[[N:.*]] = memref.dim %{{.*}}, %c1 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-002:   %[[sBj:.*]] = memref.subview %{{.*}}[%[[K]], 0] [%[[szK]], %[[N]]] [1, 1] : memref<?x?xf32, #[[$strided2D]]> to memref<?x?xf32, #[[$strided2D]]>
//       TILE-002:   linalg.matmul ins(%[[sAj]], %[[sBj]]{{.*}} outs(%{{.*}}

// TILE-234-LABEL: func @matmul(
//       TILE-234-DAG: %[[C0:.*]] = arith.constant 0 : index
//       TILE-234-DAG: %[[C2:.*]] = arith.constant 2 : index
//       TILE-234-DAG: %[[C3:.*]] = arith.constant 3 : index
//       TILE-234-DAG: %[[C4:.*]] = arith.constant 4 : index
//       TILE-234: %[[ubM:.*]] = memref.dim %{{.*}}, %c0 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-234: %[[ubK:.*]] = memref.dim %{{.*}}, %c1 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-234: %[[ubN:.*]] = memref.dim %{{.*}}, %c1 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-234:  scf.for %[[I:.*]] = %{{.*}}{{.*}} to %[[ubM]] step %{{.*}} {
//       TILE-234:    scf.for %[[J:.*]] = %{{.*}}{{.*}} to %[[ubN]] step %{{.*}} {
//       TILE-234:      scf.for %[[K:.*]] = %{{.*}}{{.*}} to %[[ubK]] step %{{.*}} {
//       TILE-234:        %[[szM:.*]] = affine.min #[[$bound_map_2]](%[[I]])[%[[ubM]]]
//       TILE-234:        %[[szK:.*]] = affine.min #[[$bound_map_4]](%[[K]])[%[[ubK]]]
//       TILE-234:        %[[sAik:.*]] = memref.subview %{{.*}}[%[[I]], %[[K]]] [%[[szM]], %[[szK]]] [1, 1] : memref<?x?xf32, #[[$strided2D]]> to memref<?x?xf32, #[[$strided2D]]>
//       TILE-234:        %[[szK:.*]] = affine.min #[[$bound_map_4]](%[[K]])[%[[ubK]]]
//       TILE-234:        %[[szN:.*]] = affine.min #[[$bound_map_3]](%[[J]])[%[[ubN]]]
//       TILE-234:        %[[sBkj:.*]] = memref.subview %{{.*}}[%[[K]], %[[J]]] [%[[szK]], %[[szN]]] [1, 1] : memref<?x?xf32, #[[$strided2D]]> to memref<?x?xf32, #[[$strided2D]]>
//       TILE-234:        %[[szM:.*]] = affine.min #[[$bound_map_2]](%[[I]])[%[[ubM]]]
//       TILE-234:        %[[szN:.*]] = affine.min #[[$bound_map_3]](%[[J]])[%[[ubN]]]
//       TILE-234:        %[[sCij:.*]] = memref.subview %{{.*}}[%[[I]], %[[J]]] [%[[szM]], %[[szN]]] [1, 1] : memref<?x?xf32, #[[$strided2D]]> to memref<?x?xf32, #[[$strided2D]]>
//
//       TILE-234:        linalg.matmul ins(%[[sAik]], %[[sBkj]]{{.*}} outs(%[[sCij]]

// When the buffer shapes are known at compile time, it is possible to avoid
// the "min" in subview size computation. This test uses buffer sizes divisible
// by respective tile sizes (M=10 divisble by 2, N=12 divisible by 2 and 3,
// K=16 divisble by 2 and 4).
func.func @matmul_static(%arg0: memref<10x16xf32, offset: ?, strides: [?, 1]>,
                    %arg1: memref<16x12xf32, offset: ?, strides: [?, 1]>,
                    %arg2: memref<10x12xf32, offset: ?, strides: [?, 1]>) {
  linalg.matmul
    ins(%arg0, %arg1: memref<10x16xf32, offset: ?, strides: [?, 1]>,
                      memref<16x12xf32, offset: ?, strides: [?, 1]>)
   outs(%arg2: memref<10x12xf32, offset: ?, strides: [?, 1]>)
  return
}
// TILE-2-LABEL: func @matmul_static(
//  TILE-2-SAME: %[[ARG0:[0-9a-zA-Z]*]]: memref
//  TILE-2-SAME: %[[ARG1:[0-9a-zA-Z]*]]: memref
//  TILE-2-SAME: %[[ARG2:[0-9a-zA-Z]*]]: memref
//       TILE-2-DAG: %[[C0:.*]] = arith.constant 0 : index
//       TILE-2-DAG: %[[C2:.*]] = arith.constant 2 : index
//       TILE-2-DAG: %[[M:.*]] = arith.constant 10 : index
//       TILE-2: scf.for %[[I:.*]] = %{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[sAi:.*]] = memref.subview %{{.*}}[%[[I]], 0] [2, 16] [1, 1] : memref<10x16xf32, #[[$strided2D]]> to memref<2x16xf32, #[[$strided2D]]>
//       TILE-2:   %[[sCi:.*]] = memref.subview %{{.*}}[%[[I]], 0] [2, 12] [1, 1] : memref<10x12xf32, #[[$strided2D]]> to memref<2x12xf32, #[[$strided2D]]>
//       TILE-2:   linalg.matmul ins(%[[sAi]], %{{.*}}{{.*}} outs(%[[sCi]]

// TILE-02-LABEL: func @matmul_static(
//       TILE-02-DAG: %[[C0:.*]] = arith.constant 0 : index
//       TILE-02-DAG: %[[C2:.*]] = arith.constant 2 : index
//       TILE-02-DAG: %[[N:.*]] = arith.constant 12 : index
//       TILE-02: scf.for %[[J:.*]] = %{{.*}} to %[[N]] step %{{.*}} {
//       TILE-02:   %[[sBj:.*]] = memref.subview %{{.*}}[0, %[[J]]] [16, 2] [1, 1] : memref<16x12xf32, #[[$strided2D]]> to memref<16x2xf32, #[[$strided2D]]>
//       TILE-02:   %[[sCj:.*]] = memref.subview %{{.*}}[0, %[[J]]] [10, 2] [1, 1] : memref<10x12xf32, #[[$strided2D]]> to memref<10x2xf32, #[[$strided2D]]>
//       TILE-02:   linalg.matmul ins(%{{.*}}, %[[sBj]]{{.*}} outs(%[[sCj]]

// TILE-002-LABEL: func @matmul_static(
//       TILE-002-DAG: %[[C0:.*]] = arith.constant 0 : index
//       TILE-002-DAG: %[[C2:.*]] = arith.constant 2 : index
//       TILE-002-DAG: %[[C16:.*]] = arith.constant 16 : index
//       TILE-002: scf.for %[[K:.*]] = %{{.*}}{{.*}} to %[[C16]] step %{{.*}} {
//       TILE-002:   %[[sAj:.*]] = memref.subview %{{.*}}[0, %[[K]]] [10, 2] [1, 1] : memref<10x16xf32, #[[$strided2D]]> to memref<10x2xf32, #[[$strided2D]]>
//       TILE-002:   %[[sBj:.*]] = memref.subview %{{.*}}[%[[K]], 0] [2, 12] [1, 1] : memref<16x12xf32, #[[$strided2D]]> to memref<2x12xf32, #[[$strided2D]]>
//       TILE-002:   linalg.matmul ins(%[[sAj]], %[[sBj]]{{.*}} outs(%{{.*}}

// TILE-234-LABEL: func @matmul_static(
//       TILE-234-DAG: %[[C0:.*]] = arith.constant 0 : index
//       TILE-234-DAG: %[[C2:.*]] = arith.constant 2 : index
//       TILE-234-DAG: %[[C3:.*]] = arith.constant 3 : index
//       TILE-234-DAG: %[[C4:.*]] = arith.constant 4 : index
//       TILE-234-DAG: %[[C10:.*]] = arith.constant 10 : index
//       TILE-234-DAG: %[[C16:.*]] = arith.constant 16 : index
//       TILE-234-DAG: %[[C12:.*]] = arith.constant 12 : index
//       TILE-234:  scf.for %[[I:.*]] = %{{.*}}{{.*}} to %[[C10]] step %{{.*}} {
//       TILE-234:    scf.for %[[J:.*]] = %{{.*}}{{.*}} to %[[C12]] step %{{.*}} {
//       TILE-234:      scf.for %[[K:.*]] = %{{.*}}{{.*}} to %[[C16]] step %{{.*}} {
//       TILE-234:        %[[sAik:.*]] = memref.subview %{{.*}}[%[[I]], %[[K]]] [2, 4] [1, 1] : memref<10x16xf32, #[[$strided2D]]> to memref<2x4xf32, #[[$strided2D]]>
//       TILE-234:        %[[sBkj:.*]] = memref.subview %{{.*}}[%[[K]], %[[J]]] [4, 3] [1, 1] : memref<16x12xf32, #[[$strided2D]]> to memref<4x3xf32, #[[$strided2D]]>
//       TILE-234:        %[[sCij:.*]] = memref.subview %{{.*}}[%[[I]], %[[J]]] [2, 3] [1, 1] : memref<10x12xf32, #[[$strided2D]]> to memref<2x3xf32, #[[$strided2D]]>
//
//       TILE-234:        linalg.matmul ins(%[[sAik]], %[[sBkj]]{{.*}} outs(%[[sCij]]

func.func @matvec(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>, %arg2: memref<?xf32, offset: ?, strides: [1]>) {
  linalg.matvec
    ins(%arg0, %arg1: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                      memref<?xf32, offset: ?, strides: [1]>)
   outs(%arg2: memref<?xf32, offset: ?, strides: [1]>)
  return
}
// TILE-2-LABEL: func @matvec(
//  TILE-2-SAME: %[[ARG0:[0-9a-zA-Z]*]]: memref
//  TILE-2-SAME: %[[ARG1:[0-9a-zA-Z]*]]: memref
//  TILE-2-SAME: %[[ARG2:[0-9a-zA-Z]*]]: memref
//       TILE-2-DAG: %[[C0:.*]] = arith.constant 0 : index
//       TILE-2-DAG: %[[C2:.*]] = arith.constant 2 : index
//       TILE-2: %[[M:.*]] = memref.dim %{{.*}}, %c0 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-2: scf.for %[[I:.*]] = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[szM:.*]] = affine.min #[[$bound_map]](%[[I]])[%[[M]]]
//       TILE-2:   %[[N:.*]] = memref.dim %{{.*}}, %c1 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-2:   %[[sAi:.*]] = memref.subview %{{.*}}[%[[I]], 0] [%[[szM]], %[[N]]] [1, 1] : memref<?x?xf32, #[[$strided2D]]> to memref<?x?xf32, #[[$strided2D]]>
//       TILE-2:   %[[szN:.*]] = affine.min #[[$bound_map]](%[[I]])[%[[M]]]
//       TILE-2:   %[[sCi:.*]] = memref.subview %{{.*}}[%[[I]]] [%[[szN]]] [1] : memref<?xf32, #[[$strided1D]]> to memref<?xf32, #[[$strided1D]]>
//       TILE-2:   linalg.matvec ins(%[[sAi]], %{{.*}} outs(%[[sCi]]

// TILE-02-LABEL: func @matvec(
// TILE-02-SAME: %[[ARG0:[0-9a-zA-Z]*]]: memref
// TILE-02-SAME: %[[ARG1:[0-9a-zA-Z]*]]: memref
// TILE-02-SAME: %[[ARG2:[0-9a-zA-Z]*]]: memref
//       TILE-02-DAG: %[[C0:.*]] = arith.constant 0 : index
//       TILE-02-DAG: %[[C2:.*]] = arith.constant 2 : index
//       TILE-02: %[[K:.*]] = memref.dim %{{.*}}, %c1 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-02: scf.for %[[J:.*]] = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-02:   %[[M:.*]] = memref.dim %{{.*}}, %c0 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-02:   %[[szN:.*]] = affine.min #[[$bound_map]](%[[J]])[%[[K]]]
//       TILE-02:   %[[sAj:.*]] = memref.subview %{{.*}}[0, %[[J]]] [%[[M]], %[[szN]]] [1, 1] : memref<?x?xf32, #[[$strided2D]]> to memref<?x?xf32, #[[$strided2D]]>
//       TILE-02:   %[[szN:.*]] = affine.min #[[$bound_map]](%[[J]])[%[[K]]]
//       TILE-02:   %[[sBj:.*]] = memref.subview %{{.*}}[%[[J]]] [%[[szN]]] [1] : memref<?xf32, #[[$strided1D]]> to memref<?xf32, #[[$strided1D]]>
//       TILE-02:   linalg.matvec ins(%[[sAj]], %[[sBj]]{{.*}} outs(%{{.*}}

// TILE-002-LABEL: func @matvec(
// TILE-002-SAME: %[[ARG0:[0-9a-zA-Z]*]]: memref
// TILE-002-SAME: %[[ARG1:[0-9a-zA-Z]*]]: memref
// TILE-002-SAME: %[[ARG2:[0-9a-zA-Z]*]]: memref
//   TILE-002-NOT: scf.for

// TILE-234-LABEL: func @matvec(
// TILE-234-SAME: %[[ARG0:[0-9a-zA-Z]*]]: memref
// TILE-234-SAME: %[[ARG1:[0-9a-zA-Z]*]]: memref
// TILE-234-SAME: %[[ARG2:[0-9a-zA-Z]*]]: memref
//       TILE-234-DAG: %[[C0:.*]] = arith.constant 0 : index
//       TILE-234-DAG: %[[C2:.*]] = arith.constant 2 : index
//       TILE-234-DAG: %[[C3:.*]] = arith.constant 3 : index
//       TILE-234: %[[M:.*]] = memref.dim %{{.*}}, %c0 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-234: %[[K:.*]] = memref.dim %{{.*}}, %c1 : memref<?x?xf32, #[[$strided2D]]>
//       TILE-234:  scf.for %[[I:.*]] = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-234:    scf.for %[[J:.*]] = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-234:      %[[szM:.*]] = affine.min #[[$bound_map_2]](%[[I]])[%[[M]]]
//       TILE-234:      %[[szN:.*]] = affine.min #[[$bound_map_3]](%[[J]])[%[[K]]]
//       TILE-234:      %[[sAij:.*]] = memref.subview %{{.*}}[%[[I]], %[[J]]] [%[[szM]], %[[szN]]] [1, 1] : memref<?x?xf32, #[[$strided2D]]> to memref<?x?xf32, #[[$strided2D]]>
//       TILE-234:      %[[szN:.*]] = affine.min #[[$bound_map_3]](%[[J]])[%[[K]]]
//       TILE-234:      %[[sBj:.*]] = memref.subview %{{.*}}[%[[J]]] [%[[szN]]] [1] : memref<?xf32, #[[$strided1D]]> to memref<?xf32, #[[$strided1D]]>
//       TILE-234:      %[[szM:.*]] = affine.min #[[$bound_map_2]](%[[I]])[%[[M]]]
//       TILE-234:      %[[sCi:.*]] = memref.subview %{{.*}}[%[[I]]] [%[[szM]]] [1] : memref<?xf32, #[[$strided1D]]> to memref<?xf32, #[[$strided1D]]>
//
//       TILE-234:      linalg.matvec ins(%[[sAij]], %[[sBj]]{{.*}} outs(%[[sCi]]

func.func @dot(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>, %arg2: memref<f32>) {
  linalg.dot
    ins(%arg0, %arg1: memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>)
   outs(%arg2: memref<f32>)
  return
}
// TILE-2-LABEL: func @dot(
//       TILE-2-DAG: %[[C0:.*]] = arith.constant 0 : index
//       TILE-2-DAG: %[[C2:.*]] = arith.constant 2 : index
//       TILE-2: %[[M:.*]] = memref.dim %{{.*}}, %c0 : memref<?xf32, #[[$strided1D]]>
//       TILE-2: scf.for %[[I:.*]] = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[szM:.*]] = affine.min #[[$bound_map]](%[[I]])[%[[M]]]
//       TILE-2:   %[[sAi:.*]] = memref.subview %{{.*}}[%[[I]]] [%[[szM]]] [1] : memref<?xf32, #[[$strided1D]]> to memref<?xf32, #[[$strided1D]]>
//       TILE-2:   %[[szM:.*]] = affine.min #[[$bound_map]](%[[I]])[%[[M]]]
//       TILE-2:   %[[sBi:.*]] = memref.subview %{{.*}}[%[[I]]] [%[[szM]]] [1] : memref<?xf32, #[[$strided1D]]> to memref<?xf32, #[[$strided1D]]>
//       TILE-2:   linalg.dot ins(%[[sAi]], %[[sBi]]{{.*}} outs(

// TILE-02-LABEL: func @dot(
//   TILE-02-NOT: scf.for

// TILE-002-LABEL: func @dot(
//   TILE-002-NOT: scf.for

// TILE-234-LABEL: func @dot(
//       TILE-234-DAG: %[[C0:.*]] = arith.constant 0 : index
//       TILE-234-DAG: %[[C2:.*]] = arith.constant 2 : index
//       TILE-234:  %[[ubK:.*]] = memref.dim %{{.*}}, %c0 : memref<?xf32, #[[$strided1D]]>
//       TILE-234:  scf.for %[[I:.*]] = %{{.*}} to %[[ubK]] step %{{.*}} {
//       TILE-234:    %[[szM:.*]] = affine.min #[[$bound_map_2]](%[[I]])[%[[ubK]]]
//       TILE-234:    %[[sAi:.*]] = memref.subview %{{.*}}[%[[I]]] [%[[szM]]] [1] : memref<?xf32, #[[$strided1D]]> to memref<?xf32, #[[$strided1D]]>
//       TILE-234:    %[[szM:.*]] = affine.min #[[$bound_map_2]](%[[I]])[%[[ubK]]]
//       TILE-234:    %[[sBi:.*]] = memref.subview %{{.*}}[%[[I]]] [%[[szM]]] [1] : memref<?xf32, #[[$strided1D]]> to memref<?xf32, #[[$strided1D]]>
//       TILE-234:    linalg.dot ins(%[[sAi]], %[[sBi]]{{.*}} outs(

func.func @fill_static(%arg0: memref<127x99xf32>, %arg1: f32) {
  linalg.fill ins(%arg1 : f32) outs(%arg0 : memref<127x99xf32>)
  return
}
// TILE-2-LABEL: func @fill_static
//       TILE-2:   for
//   TILE-2-NOT:   for
//       TILE-2:       memref.subview{{.*}} : memref<127x99xf32>
//       TILE-2:       linalg.fill{{.*}} : memref<?x99xf32, #[[$stride_99_1_layout_map]]>

// TILE-02-LABEL: func @fill_static
//       TILE-02:   for
//   TILE-02-NOT:   for
//       TILE-02:       memref.subview{{.*}} : memref<127x99xf32>
//       TILE-02:       linalg.fill{{.*}} : memref<127x?xf32, #[[$stride_99_1_layout_map]]>

// TILE-002-LABEL: func @fill_static
//   TILE-002-NOT:   for
//       TILE-002:     linalg.fill{{.*}} : memref<127x99xf32>

// TILE-234-LABEL: func @fill_static
//       TILE-234:   for
//       TILE-234:     for
//   TILE-234-NOT:   for
//       TILE-234:       memref.subview{{.*}} : memref<127x99xf32>
//       TILE-234:       linalg.fill{{.*}} : memref<?x3xf32, #[[$stride_99_1_layout_map]]>


func.func @fill(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: f32) {
  linalg.fill ins(%arg1 : f32) outs(%arg0 : memref<?x?xf32, offset: ?, strides: [?, 1]>)
  return
}
// TILE-2-LABEL: func @fill
//       TILE-2:   for
//   TILE-2-NOT:   for
//       TILE-2:   fill{{.*}} f32

// TILE-02-LABEL: func @fill
//       TILE-02:   for
//   TILE-02-NOT:   for
//       TILE-02:     fill{{.*}} f32

// TILE-002-LABEL: func @fill
//   TILE-002-NOT:   for
//       TILE-002:     fill{{.*}} f32

// TILE-234-LABEL: func @fill
//       TILE-234:   for
//       TILE-234:     for
//   TILE-234-NOT:   for
//       TILE-234:       fill{{.*}} f32

#id_2d = affine_map<(i, j) -> (i, j)>
#pointwise_2d_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  iterator_types = ["parallel", "parallel"]
}

func.func @pointwise(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                %arg2: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.generic #pointwise_2d_trait
    ins(%arg0, %arg1 : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>)
    outs(%arg2 : memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
    %4 = arith.addf %arg4, %arg5 : f32
    linalg.yield %4 : f32
  }
  return
}
// TILE-2-LABEL: func @pointwise
//       TILE-2:   for
//   TILE-2-NOT:   for
//       TILE-2:   linalg.generic

// TILE-02-LABEL: func @pointwise
//       TILE-02:   for
//   TILE-02-NOT:   for
//       TILE-02:     linalg.generic

// TILE-002-LABEL: func @pointwise
//   TILE-002-NOT:   for
//       TILE-002:     linalg.generic

// TILE-234-LABEL: func @pointwise
//       TILE-234:   for
//       TILE-234:     for
//   TILE-234-NOT:   for
//       TILE-234:       linalg.generic
