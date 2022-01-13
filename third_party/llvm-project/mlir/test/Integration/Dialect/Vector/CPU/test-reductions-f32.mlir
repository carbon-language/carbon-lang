// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @entry() {
  // Construct test vector.
  %f1 = constant 1.5: f32
  %f2 = constant 2.0: f32
  %f3 = constant 3.0: f32
  %f4 = constant 4.0: f32
  %f5 = constant 5.0: f32
  %f6 = constant -1.0: f32
  %f7 = constant -2.0: f32
  %f8 = constant -4.0: f32
  %f9 = constant -0.25: f32
  %f10 = constant -16.0: f32
  %v0 = vector.broadcast %f1 : f32 to vector<10xf32>
  %v1 = vector.insert %f2, %v0[1] : f32 into vector<10xf32>
  %v2 = vector.insert %f3, %v1[2] : f32 into vector<10xf32>
  %v3 = vector.insert %f4, %v2[3] : f32 into vector<10xf32>
  %v4 = vector.insert %f5, %v3[4] : f32 into vector<10xf32>
  %v5 = vector.insert %f6, %v4[5] : f32 into vector<10xf32>
  %v6 = vector.insert %f7, %v5[6] : f32 into vector<10xf32>
  %v7 = vector.insert %f8, %v6[7] : f32 into vector<10xf32>
  %v8 = vector.insert %f9, %v7[8] : f32 into vector<10xf32>
  %v9 = vector.insert %f10, %v8[9] : f32 into vector<10xf32>
  vector.print %v9 : vector<10xf32>
  //
  // test vector:
  //
  // CHECK: ( 1.5, 2, 3, 4, 5, -1, -2, -4, -0.25, -16 )

  // Various vector reductions. Not full functional unit tests, but
  // a simple integration test to see if the code runs end-to-end.
  %0 = vector.reduction "add", %v9 : vector<10xf32> into f32
  vector.print %0 : f32
  // CHECK: -7.75
  %1 = vector.reduction "mul", %v9 : vector<10xf32> into f32
  vector.print %1 : f32
  // CHECK: -5760
  %2 = vector.reduction "min", %v9 : vector<10xf32> into f32
  vector.print %2 : f32
  // CHECK: -16
  %3 = vector.reduction "max", %v9 : vector<10xf32> into f32
  vector.print %3 : f32
  // CHECK: 5

  return
}
