// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @entry() {
  %v = std.constant dense<[-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<24xi4>
  vector.print %v : vector<24xi4>
  //
  // Test vector:
  //
  // CHECK: ( -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1 )


  %0 = vector.reduction "add", %v : vector<24xi4> into i4
  vector.print %0 : i4
  // CHECK: 4

  %1 = vector.reduction "mul", %v : vector<24xi4> into i4
  vector.print %1 : i4
  // CHECK: 0

  %2 = vector.reduction "min", %v : vector<24xi4> into i4
  vector.print %2 : i4
  // CHECK: -8

  %3 = vector.reduction "max", %v : vector<24xi4> into i4
  vector.print %3 : i4
  // CHECK: 7

  %4 = vector.reduction "and", %v : vector<24xi4> into i4
  vector.print %4 : i4
  // CHECK: 0

  %5 = vector.reduction "or", %v : vector<24xi4> into i4
  vector.print %5 : i4
  // CHECK: -1

  %6 = vector.reduction "xor", %v : vector<24xi4> into i4
  vector.print %6 : i4
  // CHECK: 0

  return
}
