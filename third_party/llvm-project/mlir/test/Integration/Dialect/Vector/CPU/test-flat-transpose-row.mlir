// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -O0 -enable-matrix -matrix-allow-contract -matrix-default-layout=row-major \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func.func @entry() {
  %f0 = arith.constant 0.0: f64
  %f1 = arith.constant 1.0: f64
  %f2 = arith.constant 2.0: f64
  %f3 = arith.constant 3.0: f64
  %f4 = arith.constant 4.0: f64
  %f5 = arith.constant 5.0: f64
  %f6 = arith.constant 6.0: f64
  %f7 = arith.constant 7.0: f64

  // Construct test vectors.
  %0 = vector.broadcast %f0 : f64 to vector<4xf64>
  %1 = vector.insert %f1, %0[1] : f64 into vector<4xf64>
  %2 = vector.insert %f2, %1[2] : f64 into vector<4xf64>
  %a = vector.insert %f3, %2[3] : f64 into vector<4xf64>
  %3 = vector.broadcast %f4 : f64 to vector<4xf64>
  %4 = vector.insert %f5, %3[1] : f64 into vector<4xf64>
  %5 = vector.insert %f6, %4[2] : f64 into vector<4xf64>
  %b = vector.insert %f7, %5[3] : f64 into vector<4xf64>
  %6 = vector.broadcast %f0 : f64 to vector<6xf64>
  %7 = vector.insert %f1, %6[1] : f64 into vector<6xf64>
  %8 = vector.insert %f2, %7[2] : f64 into vector<6xf64>
  %9 = vector.insert %f3, %8[3] : f64 into vector<6xf64>
  %10 = vector.insert %f4, %9[4] : f64 into vector<6xf64>
  %c = vector.insert %f5, %10[5] : f64 into vector<6xf64>

  vector.print %a : vector<4xf64>
  vector.print %b : vector<4xf64>
  vector.print %c : vector<6xf64>
  //
  // Test vectors:
  //
  // CHECK: ( 0, 1, 2, 3 )
  // CHECK: ( 4, 5, 6, 7 )
  // CHECK: ( 0, 1, 2, 3, 4, 5 )

  // Performs matrix transpositions interpreting the vectors as
  // flattened row-major 2-D matrices.
  //
  // ( 0, 1 )       ( 0, 2 )
  // ( 2, 3 )    -> ( 1, 3 )
  //
  // ( 4, 5 )       ( 4, 6 )
  // ( 6, 7 )    -> ( 5, 7 )
  //
  // ( 0, 1, 2 )    ( 0, 3 )
  // ( 3, 4, 5 ) -> ( 1, 4 )
  //                ( 2, 5 )
  //
  // ( 0, 1 )        ( 0, 2, 4 )
  // ( 2, 3 )    ->  ( 1, 3, 5 )
  // ( 4, 5 )
  //
  %d = vector.flat_transpose %a { rows = 2: i32, columns = 2: i32 } : vector<4xf64> -> vector<4xf64>
  %e = vector.flat_transpose %b { rows = 2: i32, columns = 2: i32 } : vector<4xf64> -> vector<4xf64>
  %f = vector.flat_transpose %c { rows = 2: i32, columns = 3: i32 } : vector<6xf64> -> vector<6xf64>
  %g = vector.flat_transpose %c { rows = 3: i32, columns = 2: i32 } : vector<6xf64> -> vector<6xf64>

  vector.print %d : vector<4xf64>
  vector.print %e : vector<4xf64>
  vector.print %f : vector<6xf64>
  vector.print %g : vector<6xf64>
  //
  // Transposed results:
  //
  // CHECK: ( 0, 2, 1, 3 )
  // CHECK: ( 4, 6, 5, 7 )
  // CHECK: ( 0, 3, 1, 4, 2, 5 )
  // CHECK: ( 0, 2, 4, 1, 3, 5 )

  return
}
