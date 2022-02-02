// RUN: mlir-opt -allow-unregistered-dialect -split-input-file %s | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s -mlir-print-op-generic | FileCheck -check-prefix=GENERIC %s

// Check that the attributes for the affine operations are round-tripped.
// Check that `affine.yield` is visible in the generic form.
// CHECK-LABEL: @empty
func @empty() {
  // CHECK: affine.for
  // CHECK-NEXT: } {some_attr = true}
  //
  // GENERIC:      "affine.for"()
  // GENERIC-NEXT: ^bb0(%{{.*}}: index):
  // GENERIC-NEXT:   "affine.yield"() : () -> ()
  // GENERIC-NEXT: })
  affine.for %i = 0 to 10 {
  } {some_attr = true}

  // CHECK: affine.if
  // CHECK-NEXT: } {some_attr = true}
  //
  // GENERIC:      "affine.if"()
  // GENERIC-NEXT:   "affine.yield"() : () -> ()
  // GENERIC-NEXT: },  {
  // GENERIC-NEXT: })
  affine.if affine_set<() : ()> () {
  } {some_attr = true}

  // CHECK: } else {
  // CHECK: } {some_attr = true}
  //
  // GENERIC:      "affine.if"()
  // GENERIC-NEXT:   "affine.yield"() : () -> ()
  // GENERIC-NEXT: },  {
  // GENERIC-NEXT:   "foo"() : () -> ()
  // GENERIC-NEXT:   "affine.yield"() : () -> ()
  // GENERIC-NEXT: })
  affine.if affine_set<() : ()> () {
  } else {
    "foo"() : () -> ()
  } {some_attr = true}

  return
}

// Check that an explicit affine.yield is not printed in custom format.
// Check that no extra terminator is introduced.
// CHECK-LABEL: @affine.yield
func @affine.yield() {
  // CHECK: affine.for
  // CHECK-NEXT: }
  //
  // GENERIC:      "affine.for"() ( {
  // GENERIC-NEXT: ^bb0(%{{.*}}: index):	// no predecessors
  // GENERIC-NEXT:   "affine.yield"() : () -> ()
  // GENERIC-NEXT: }) {lower_bound = #map0, step = 1 : index, upper_bound = #map1} : () -> ()
  affine.for %i = 0 to 10 {
    "affine.yield"() : () -> ()
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP0:map[0-9]+]] = affine_map<(d0)[s0] -> (1000, d0 + 512, s0)>
// CHECK-DAG: #[[$MAP1:map[0-9]+]] = affine_map<(d0, d1)[s0] -> (d0 - d1, s0 + 512)>
// CHECK-DAG: #[[$MAP2:map[0-9]+]] = affine_map<()[s0, s1] -> (s0 - s1, 11)>
// CHECK-DAG: #[[$MAP3:map[0-9]+]] = affine_map<() -> (77, 78, 79)>

// CHECK-LABEL: @affine_min
func @affine_min(%arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK: affine.min #[[$MAP0]](%arg0)[%arg1]
  %0 = affine.min affine_map<(d0)[s0] -> (1000, d0 + 512, s0)> (%arg0)[%arg1]
  // CHECK: affine.min #[[$MAP1]](%arg0, %arg1)[%arg2]
  %1 = affine.min affine_map<(d0, d1)[s0] -> (d0 - d1, s0 + 512)> (%arg0, %arg1)[%arg2]
  // CHECK: affine.min #[[$MAP2]]()[%arg1, %arg2]
  %2 = affine.min affine_map<()[s0, s1] -> (s0 - s1, 11)> ()[%arg1, %arg2]
  // CHECK: affine.min #[[$MAP3]]()
  %3 = affine.min affine_map<()[] -> (77, 78, 79)> ()[]
  return
}

// CHECK-LABEL: @affine_max
func @affine_max(%arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK: affine.max #[[$MAP0]](%arg0)[%arg1]
  %0 = affine.max affine_map<(d0)[s0] -> (1000, d0 + 512, s0)> (%arg0)[%arg1]
  // CHECK: affine.max #[[$MAP1]](%arg0, %arg1)[%arg2]
  %1 = affine.max affine_map<(d0, d1)[s0] -> (d0 - d1, s0 + 512)> (%arg0, %arg1)[%arg2]
  // CHECK: affine.max #[[$MAP2]]()[%arg1, %arg2]
  %2 = affine.max affine_map<()[s0, s1] -> (s0 - s1, 11)> ()[%arg1, %arg2]
  // CHECK: affine.max #[[$MAP3]]()
  %3 = affine.max affine_map<()[] -> (77, 78, 79)> ()[]
  return
}

// -----

func @valid_symbols(%arg0: index, %arg1: index, %arg2: index) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  affine.for %arg3 = 0 to %arg2 step 768 {
    %13 = memref.dim %0, %c1 : memref<?x?xf32>
    affine.for %arg4 = 0 to %13 step 264 {
      %18 = memref.dim %0, %c0 : memref<?x?xf32>
      %20 = memref.subview %0[%c0, %c0][%18,%arg4][%c1,%c1] : memref<?x?xf32>
                          to memref<?x?xf32, offset : ?, strides : [?, ?]>
      %24 = memref.dim %20, %c0 : memref<?x?xf32, offset : ?, strides : [?, ?]>
      affine.for %arg5 = 0 to %24 step 768 {
        "foo"() : () -> ()
      }
    }
  }
  return
}

// -----

// Test symbol constraints for ops with AffineScope trait.

// CHECK-LABEL: func @valid_symbol_affine_scope
func @valid_symbol_affine_scope(%n : index, %A : memref<?xf32>) {
  test.affine_scope {
    %c1 = arith.constant 1 : index
    %l = arith.subi %n, %c1 : index
    // %l, %n are valid symbols since test.affine_scope defines a new affine
    // scope.
    affine.for %i = %l to %n {
      %m = arith.subi %l, %i : index
      test.affine_scope {
        // %m and %n are valid symbols.
        affine.for %j = %m to %n {
          %v = affine.load %A[%n - 1] : memref<?xf32>
          affine.store %v, %A[%n - 1] : memref<?xf32>
        }
        "terminate"() : () -> ()
      }
    }
    "terminate"() : () -> ()
  }
  return
}

// -----

// Test the fact that module op always provides an affine scope.

%idx = "test.foo"() : () -> (index)
"test.func"() ({
^bb0(%A : memref<?xf32>):
  affine.load %A[%idx] : memref<?xf32>
  "terminate"() : () -> ()
}) : () -> ()

// -----

// CHECK-LABEL: func @parallel
// CHECK-SAME: (%[[A:.*]]: memref<100x100xf32>, %[[N:.*]]: index)
func @parallel(%A : memref<100x100xf32>, %N : index) {
  // CHECK: affine.parallel (%[[I0:.*]], %[[J0:.*]]) = (0, 0) to (symbol(%[[N]]), 100) step (10, 10)
  affine.parallel (%i0, %j0) = (0, 0) to (symbol(%N), 100) step (10, 10) {
    // CHECK: affine.parallel (%{{.*}}, %{{.*}}) = (%[[I0]], %[[J0]]) to (%[[I0]] + 10, %[[J0]] + 10) reduce ("minf", "maxf") -> (f32, f32)
    %0:2 = affine.parallel (%i1, %j1) = (%i0, %j0) to (%i0 + 10, %j0 + 10) reduce ("minf", "maxf") -> (f32, f32) {
      %2 = affine.load %A[%i0 + %i0, %j0 + %j1] : memref<100x100xf32>
      affine.yield %2, %2 : f32, f32
    }
  }
  return
}

// -----

// CHECK-LABEL: @parallel_min_max
// CHECK: %[[A:.*]]: index, %[[B:.*]]: index, %[[C:.*]]: index, %[[D:.*]]: index
func @parallel_min_max(%a: index, %b: index, %c: index, %d: index) {
  // CHECK: affine.parallel (%{{.*}}, %{{.*}}, %{{.*}}) =
  // CHECK:                 (max(%[[A]], %[[B]])
  // CHECK:              to (%[[C]], min(%[[C]], %[[D]]), %[[B]])
  affine.parallel (%i, %j, %k) = (max(%a, %b), %b, max(%a, %c))
                              to (%c, min(%c, %d), %b) {
    affine.yield
  }
  return
}

// -----

// CHECK-LABEL: @parallel_no_ivs
func @parallel_no_ivs() {
  // CHECK: affine.parallel () = () to ()
  affine.parallel () = () to () {
    affine.yield
  }
  return
}

// -----

// CHECK-LABEL: func @affine_if
func @affine_if() -> f32 {
  // CHECK: %[[ZERO:.*]] = arith.constant {{.*}} : f32
  %zero = arith.constant 0.0 : f32
  // CHECK: %[[OUT:.*]] = affine.if {{.*}}() -> f32 {
  %0 = affine.if affine_set<() : ()> () -> f32 {
    // CHECK: affine.yield %[[ZERO]] : f32
    affine.yield %zero : f32
  } else {
    // CHECK: affine.yield %[[ZERO]] : f32
    affine.yield %zero : f32
  }
  // CHECK: return %[[OUT]] : f32
  return %0 : f32
}

// -----

//  Test affine.for with yield values.

#set = affine_set<(d0): (d0 - 10 >= 0)>

// CHECK-LABEL: func @yield_loop
func @yield_loop(%buffer: memref<1024xf32>) -> f32 {
  %sum_init_0 = arith.constant 0.0 : f32
  %res = affine.for %i = 0 to 10 step 2 iter_args(%sum_iter = %sum_init_0) -> f32 {
    %t = affine.load %buffer[%i] : memref<1024xf32>
    %sum_next = affine.if #set(%i) -> (f32) {
      %new_sum = arith.addf %sum_iter, %t : f32
      affine.yield %new_sum : f32
    } else {
      affine.yield %sum_iter : f32
    }
    affine.yield %sum_next : f32
  }
  return %res : f32
}
// CHECK:      %[[const_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[output:.*]] = affine.for %{{.*}} = 0 to 10 step 2 iter_args(%{{.*}} = %[[const_0]]) -> (f32) {
// CHECK:        affine.if #set(%{{.*}}) -> f32 {
// CHECK:          affine.yield %{{.*}} : f32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     affine.yield %{{.*}} : f32
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.yield %{{.*}} : f32
// CHECK-NEXT: }
// CHECK-NEXT: return %[[output]] : f32

// CHECK-LABEL: func @affine_for_multiple_yield
func @affine_for_multiple_yield(%buffer: memref<1024xf32>) -> (f32, f32) {
  %init_0 = arith.constant 0.0 : f32
  %res1, %res2 = affine.for %i = 0 to 10 step 2 iter_args(%iter_arg1 = %init_0, %iter_arg2 = %init_0) -> (f32, f32) {
    %t = affine.load %buffer[%i] : memref<1024xf32>
    %ret1 = arith.addf %t, %iter_arg1 : f32
    %ret2 = arith.addf %t, %iter_arg2 : f32
    affine.yield %ret1, %ret2 : f32, f32
  }
  return %res1, %res2 : f32, f32
}
// CHECK:      %[[const_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[output:[0-9]+]]:2 = affine.for %{{.*}} = 0 to 10 step 2 iter_args(%[[iter_arg1:.*]] = %[[const_0]], %[[iter_arg2:.*]] = %[[const_0]]) -> (f32, f32) {
// CHECK:        %[[res1:.*]] = arith.addf %{{.*}}, %[[iter_arg1]] : f32
// CHECK-NEXT:   %[[res2:.*]] = arith.addf %{{.*}}, %[[iter_arg2]] : f32
// CHECK-NEXT:   affine.yield %[[res1]], %[[res2]] : f32, f32
// CHECK-NEXT: }
