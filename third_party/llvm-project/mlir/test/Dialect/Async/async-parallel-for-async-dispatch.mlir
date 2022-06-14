// RUN: mlir-opt %s -split-input-file -async-parallel-for=async-dispatch=true  \
// RUN: | FileCheck %s --dump-input=always

// CHECK-LABEL: @loop_1d(
// CHECK-SAME:    %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index
func.func @loop_1d(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<?xf32>) {
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index

  // CHECK:      %[[RANGE:.*]] = arith.subi %[[UB]], %[[LB]]
  // CHECK:      %[[TRIP_CNT:.*]] = arith.ceildivsi %[[RANGE]], %[[STEP]]
  // CHECK:      %[[IS_NOOP:.*]] = arith.cmpi eq, %[[TRIP_CNT]], %[[C0]] : index

  // CHECK:      scf.if %[[IS_NOOP]] {
  // CHECK-NEXT: } else {
  // CHECK:        scf.if {{.*}} {
  // CHECK:          call @parallel_compute_fn(%[[C0]]
  // CHECK:        } else {
  // CHECK:          %[[GROUP:.*]] = async.create_group
  // CHECK:          call @async_dispatch_fn
  // CHECK:          async.await_all %[[GROUP]]
  // CHECK:        }
  // CHECK:      }
  scf.parallel (%i) = (%arg0) to (%arg1) step (%arg2) {
    %one = arith.constant 1.0 : f32
    memref.store %one, %arg3[%i] : memref<?xf32>
  }
  return
}

// CHECK-LABEL: func private @parallel_compute_fn
// CHECK:       scf.for
// CHECK:         memref.store

// CHECK-LABEL: func private @async_dispatch_fn
// CHECK-SAME:  (
// CHECK-SAME:    %[[GROUP:arg0]]: !async.group,
// CHECK-SAME:    %[[BLOCK_START:arg1]]: index
// CHECK-SAME:    %[[BLOCK_END:arg2]]: index
// CHECK-SAME:  )
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         scf.while (%[[S0:.*]] = %[[BLOCK_START]],
// CHECK-SAME:               %[[E0:.*]] = %[[BLOCK_END]])
// While loop `before` block decides if we need to dispatch more tasks.
// CHECK:         {
// CHECK:           %[[DIFF0:.*]] = arith.subi %[[E0]], %[[S0]]
// CHECK:           %[[COND:.*]] = arith.cmpi sgt, %[[DIFF0]], %[[C1]]
// CHECK:           scf.condition(%[[COND]])
// While loop `after` block splits the range in half and submits async task
// to process the second half using the call to the same dispatch function.
// CHECK:         } do {
// CHECK:         ^bb0(%[[S1:.*]]: index, %[[E1:.*]]: index):
// CHECK:           %[[DIFF1:.*]] = arith.subi %[[E1]], %[[S1]]
// CHECK:           %[[HALF:.*]] = arith.divsi %[[DIFF1]], %[[C2]]
// CHECK:           %[[MID:.*]] = arith.addi %[[S1]], %[[HALF]]
// CHECK:           %[[TOKEN:.*]] = async.execute
// CHECK:             call @async_dispatch_fn
// CHECK:           async.add_to_group
// CHECK:           scf.yield %[[S1]], %[[MID]]
// CHECK:         }
// After async dispatch the first block processed in the caller thread.
// CHECK:         call @parallel_compute_fn(%[[BLOCK_START]]

// -----

// CHECK-LABEL: @loop_2d
func.func @loop_2d(%arg0: index, %arg1: index, %arg2: index, // lb, ub, step
              %arg3: index, %arg4: index, %arg5: index, // lb, ub, step
              %arg6: memref<?x?xf32>) {
  // CHECK: %[[GROUP:.*]] = async.create_group
  // CHECK: call @async_dispatch_fn
  // CHECK: async.await_all %[[GROUP]]
  scf.parallel (%i0, %i1) = (%arg0, %arg3) to (%arg1, %arg4)
                            step (%arg2, %arg5) {
    %one = arith.constant 1.0 : f32
    memref.store %one, %arg6[%i0, %i1] : memref<?x?xf32>
  }
  return
}

// CHECK-LABEL: func private @parallel_compute_fn
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           memref.store

// CHECK-LABEL: func private @async_dispatch_fn
