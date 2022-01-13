// RUN: mlir-opt %s | mlir-opt | FileCheck %s

func @omp_barrier() -> () {
  // CHECK: omp.barrier
  omp.barrier
  return
}

func @omp_master() -> () {
  // CHECK: omp.master
  omp.master {
    // CHECK: omp.terminator
    omp.terminator
  }

  return
}

func @omp_taskwait() -> () {
  // CHECK: omp.taskwait
  omp.taskwait
  return
}

func @omp_taskyield() -> () {
  // CHECK: omp.taskyield
  omp.taskyield
  return
}

// CHECK-LABEL: func @omp_flush
// CHECK-SAME: ([[ARG0:%.*]]: i32) {
func @omp_flush(%arg0 : i32) -> () {
  // Test without data var
  // CHECK: omp.flush
  omp.flush

  // Test with one data var
  // CHECK: omp.flush([[ARG0]] : i32)
  omp.flush(%arg0 : i32)

  // Test with two data var
  // CHECK: omp.flush([[ARG0]], [[ARG0]] : i32, i32)
  omp.flush(%arg0, %arg0: i32, i32)

  return
}

func @omp_terminator() -> () {
  // CHECK: omp.terminator
  omp.terminator
}

func @omp_parallel(%data_var : memref<i32>, %if_cond : i1, %num_threads : si32) -> () {
  // CHECK: omp.parallel if(%{{.*}}) num_threads(%{{.*}} : si32) private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) shared(%{{.*}} : memref<i32>) copyin(%{{.*}} : memref<i32>) allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
  "omp.parallel" (%if_cond, %num_threads, %data_var, %data_var, %data_var, %data_var, %data_var, %data_var) ({

  // test without if condition
  // CHECK: omp.parallel num_threads(%{{.*}} : si32) private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) shared(%{{.*}} : memref<i32>) copyin(%{{.*}} : memref<i32>) allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
    "omp.parallel"(%num_threads, %data_var, %data_var, %data_var, %data_var, %data_var, %data_var) ({
      omp.terminator
    }) {operand_segment_sizes = dense<[0,1,1,1,1,1,1,1]>: vector<8xi32>, default_val = "defshared"} : (si32, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>) -> ()

  // CHECK: omp.barrier
    omp.barrier

  // test without num_threads
  // CHECK: omp.parallel if(%{{.*}}) private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) shared(%{{.*}} : memref<i32>) copyin(%{{.*}} : memref<i32>) allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
    "omp.parallel"(%if_cond, %data_var, %data_var, %data_var, %data_var, %data_var, %data_var) ({
      omp.terminator
    }) {operand_segment_sizes = dense<[1,0,1,1,1,1,1,1]> : vector<8xi32>} : (i1, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>) -> ()

  // test without allocate
  // CHECK: omp.parallel if(%{{.*}}) num_threads(%{{.*}} : si32) private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) shared(%{{.*}} : memref<i32>) copyin(%{{.*}} : memref<i32>)
    "omp.parallel"(%if_cond, %num_threads, %data_var, %data_var, %data_var, %data_var) ({
      omp.terminator
    }) {operand_segment_sizes = dense<[1,1,1,1,1,1,0,0]> : vector<8xi32>} : (i1, si32, memref<i32>, memref<i32>, memref<i32>, memref<i32>) -> ()

    omp.terminator
  }) {operand_segment_sizes = dense<[1,1,1,1,1,1,1,1]> : vector<8xi32>, proc_bind_val = "spread"} : (i1, si32, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>) -> ()

  // test with multiple parameters for single variadic argument
  // CHECK: omp.parallel private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>, %{{.*}} : memref<i32>) shared(%{{.*}} : memref<i32>) copyin(%{{.*}} : memref<i32>) allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
  "omp.parallel" (%data_var, %data_var, %data_var, %data_var, %data_var, %data_var, %data_var) ({
    omp.terminator
  }) {operand_segment_sizes = dense<[0,0,1,2,1,1,1,1]> : vector<8xi32>} : (memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>) -> ()

  return
}

func @omp_parallel_pretty(%data_var : memref<i32>, %if_cond : i1, %num_threads : si32, %allocator : si32) -> () {
 // CHECK: omp.parallel
 omp.parallel {
  omp.terminator
 }

 // CHECK: omp.parallel num_threads(%{{.*}} : si32)
 omp.parallel num_threads(%num_threads : si32) {
   omp.terminator
 }

 // CHECK: omp.parallel allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
 omp.parallel allocate(%data_var : memref<i32> -> %data_var : memref<i32>) {
   omp.terminator
 }

 // CHECK: omp.parallel private(%{{.*}} : memref<i32>, %{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>)
 omp.parallel private(%data_var : memref<i32>, %data_var : memref<i32>) firstprivate(%data_var : memref<i32>) {
   omp.terminator
 }

 // CHECK omp.parallel shared(%{{.*}} : memref<i32>) copyin(%{{.*}} : memref<i32>, %{{.*}} : memref<i32>)
 omp.parallel shared(%data_var : memref<i32>) copyin(%data_var : memref<i32>, %data_var : memref<i32>) {
   omp.parallel if(%if_cond: i1) {
     omp.terminator
   }
   omp.terminator
 }

 // CHECK omp.parallel if(%{{.*}}) num_threads(%{{.*}} : si32) private(%{{.*}} : memref<i32>) proc_bind(close)
 omp.parallel num_threads(%num_threads : si32) if(%if_cond: i1)
              private(%data_var : memref<i32>) proc_bind(close) {
   omp.terminator
 }

  // CHECK: omp.parallel default(private)
  omp.parallel default(private) {
    omp.terminator
  }

  // CHECK: omp.parallel default(firstprivate)
  omp.parallel default(firstprivate) {
    omp.terminator
  }

  // CHECK: omp.parallel default(shared)
  omp.parallel default(shared) {
    omp.terminator
  }

  // CHECK: omp.parallel default(none)
  omp.parallel default(none) {
    omp.terminator
  }

  return
}

// CHECK-LABEL: omp_wsloop
func @omp_wsloop(%lb : index, %ub : index, %step : index, %data_var : memref<i32>, %linear_var : i32, %chunk_var : i32) -> () {

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) private(%{{.*}} : memref<i32>, %{{.*}} : memref<i32>) collapse(2) ordered(1)
  "omp.wsloop" (%lb, %ub, %step, %data_var, %data_var) ({
    ^bb0(%iv: index):
      omp.yield
  }) {operand_segment_sizes = dense<[1,1,1,2,0,0,0,0,0,0]> : vector<10xi32>, collapse_val = 2, ordered_val = 1} :
    (index, index, index, memref<i32>, memref<i32>) -> ()

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) linear(%{{.*}} = %{{.*}} : memref<i32>) schedule(static)
  "omp.wsloop" (%lb, %ub, %step, %data_var, %linear_var) ({
    ^bb0(%iv: index):
      omp.yield
  }) {operand_segment_sizes = dense<[1,1,1,0,0,0,1,1,0,0]> : vector<10xi32>, schedule_val = "Static"} :
    (index, index, index, memref<i32>, i32) -> ()

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) linear(%{{.*}} = %{{.*}} : memref<i32>, %{{.*}} = %{{.*}} : memref<i32>) schedule(static)
  "omp.wsloop" (%lb, %ub, %step, %data_var, %data_var, %linear_var, %linear_var) ({
    ^bb0(%iv: index):
      omp.yield
  }) {operand_segment_sizes = dense<[1,1,1,0,0,0,2,2,0,0]> : vector<10xi32>, schedule_val = "Static"} :
    (index, index, index, memref<i32>, memref<i32>, i32, i32) -> ()

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) lastprivate(%{{.*}} : memref<i32>) linear(%{{.*}} = %{{.*}} : memref<i32>) schedule(dynamic = %{{.*}}) collapse(3) ordered(2)
  "omp.wsloop" (%lb, %ub, %step, %data_var, %data_var, %data_var, %data_var, %linear_var, %chunk_var) ({
    ^bb0(%iv: index):
      omp.yield
  }) {operand_segment_sizes = dense<[1,1,1,1,1,1,1,1,0,1]> : vector<10xi32>, schedule_val = "Dynamic", collapse_val = 3, ordered_val = 2} :
    (index, index, index, memref<i32>, memref<i32>, memref<i32>, memref<i32>, i32, i32) -> ()

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) private(%{{.*}} : memref<i32>) schedule(auto) nowait
  "omp.wsloop" (%lb, %ub, %step, %data_var) ({
    ^bb0(%iv: index):
      omp.yield
  }) {operand_segment_sizes = dense<[1,1,1,1,0,0,0,0,0,0]> : vector<10xi32>, nowait, schedule_val = "Auto"} :
    (index, index, index, memref<i32>) -> ()

  return
}

// CHECK-LABEL: omp_wsloop_pretty
func @omp_wsloop_pretty(%lb : index, %ub : index, %step : index,
                 %data_var : memref<i32>, %linear_var : i32, %chunk_var : i32) -> () {

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) private(%{{.*}} : memref<i32>)
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) private(%data_var : memref<i32>) collapse(2) ordered(2) {
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) linear(%{{.*}} = %{{.*}} : memref<i32>) schedule(static)
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) schedule(static) lastprivate(%data_var : memref<i32>) linear(%data_var = %linear_var : memref<i32>) {
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) lastprivate(%{{.*}} : memref<i32>) linear(%{{.*}} = %{{.*}} : memref<i32>) schedule(static = %{{.*}}) collapse(3) ordered(2)
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) ordered(2) private(%data_var : memref<i32>)
     firstprivate(%data_var : memref<i32>) lastprivate(%data_var : memref<i32>) linear(%data_var = %linear_var : memref<i32>)
     schedule(static = %chunk_var) collapse(3) {
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) lastprivate(%{{.*}} : memref<i32>) linear(%{{.*}} = %{{.*}} : memref<i32>) schedule(dynamic = %{{.*}}, nonmonotonic) collapse(3) ordered(2)
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) ordered(2) private(%data_var : memref<i32>)
     firstprivate(%data_var : memref<i32>) lastprivate(%data_var : memref<i32>) linear(%data_var = %linear_var : memref<i32>)
     schedule(dynamic = %chunk_var, nonmonotonic) collapse(3) {
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) lastprivate(%{{.*}} : memref<i32>) linear(%{{.*}} = %{{.*}} : memref<i32>) schedule(dynamic = %{{.*}}, monotonic) collapse(3) ordered(2)
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) ordered(2) private(%data_var : memref<i32>)
     firstprivate(%data_var : memref<i32>) lastprivate(%data_var : memref<i32>) linear(%data_var = %linear_var : memref<i32>)
     schedule(dynamic = %chunk_var, monotonic) collapse(3) {
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) private({{.*}} : memref<i32>)
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) private(%data_var : memref<i32>) {
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}})
  omp.wsloop (%iv) : index = (%lb) to (%ub) inclusive step (%step) {
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) nowait
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) nowait {
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) nowait order(concurrent)
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) order(concurrent) nowait {
    omp.yield
  }

  return
}

// CHECK-LABEL: omp_wsloop_pretty_multi_block
func @omp_wsloop_pretty_multi_block(%lb : index, %ub : index, %step : index, %data1 : memref<?xi32>, %data2 : memref<?xi32>) -> () {

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) {
    %1 = "test.payload"(%iv) : (index) -> (i32)
    br ^bb1(%1: i32)
  ^bb1(%arg: i32):
    memref.store %arg, %data1[%iv] : memref<?xi32>
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) {
    %c = "test.condition"(%iv) : (index) -> (i1)
    %v1 = "test.payload"(%iv) : (index) -> (i32)
    cond_br %c, ^bb1(%v1: i32), ^bb2(%v1: i32)
  ^bb1(%arg0: i32):
    memref.store %arg0, %data1[%iv] : memref<?xi32>
    br ^bb3
  ^bb2(%arg1: i32):
    memref.store %arg1, %data2[%iv] : memref<?xi32>
    br ^bb3
  ^bb3:
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) {
    %c = "test.condition"(%iv) : (index) -> (i1)
    %v1 = "test.payload"(%iv) : (index) -> (i32)
    cond_br %c, ^bb1(%v1: i32), ^bb2(%v1: i32)
  ^bb1(%arg0: i32):
    memref.store %arg0, %data1[%iv] : memref<?xi32>
    omp.yield
  ^bb2(%arg1: i32):
    memref.store %arg1, %data2[%iv] : memref<?xi32>
    omp.yield
  }

  return
}

// CHECK-LABEL: omp_wsloop_pretty_non_index
func @omp_wsloop_pretty_non_index(%lb1 : i32, %ub1 : i32, %step1 : i32, %lb2 : i64, %ub2 : i64, %step2 : i64,
                           %data1 : memref<?xi32>, %data2 : memref<?xi64>) -> () {

  // CHECK: omp.wsloop (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
  omp.wsloop (%iv1) : i32 = (%lb1) to (%ub1) step (%step1) {
    %1 = "test.payload"(%iv1) : (i32) -> (index)
    br ^bb1(%1: index)
  ^bb1(%arg1: index):
    memref.store %iv1, %data1[%arg1] : memref<?xi32>
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : i64 = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
  omp.wsloop (%iv2) : i64 = (%lb2) to (%ub2) step (%step2) {
    %2 = "test.payload"(%iv2) : (i64) -> (index)
    br ^bb1(%2: index)
  ^bb1(%arg2: index):
    memref.store %iv2, %data2[%arg2] : memref<?xi64>
    omp.yield
  }

  return
}

// CHECK-LABEL: omp_wsloop_pretty_multiple
func @omp_wsloop_pretty_multiple(%lb1 : i32, %ub1 : i32, %step1 : i32, %lb2 : i32, %ub2 : i32, %step2 : i32, %data1 : memref<?xi32>) -> () {

  // CHECK: omp.wsloop (%{{.*}}, %{{.*}}) : i32 = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}})
  omp.wsloop (%iv1, %iv2) : i32 = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    %1 = "test.payload"(%iv1) : (i32) -> (index)
    %2 = "test.payload"(%iv2) : (i32) -> (index)
    memref.store %iv1, %data1[%1] : memref<?xi32>
    memref.store %iv2, %data1[%2] : memref<?xi32>
    omp.yield
  }

  return
}

// CHECK-LABEL: omp_target
func @omp_target(%if_cond : i1, %device : si32,  %num_threads : si32) -> () {

    // Test with optional operands; if_expr, device, thread_limit, and nowait.
    // CHECK: omp.target
    "omp.target"(%if_cond, %device, %num_threads) ({
       // CHECK: omp.terminator
       omp.terminator
    }) {operand_segment_sizes = dense<[1,1,1]>: vector<3xi32>, nowait } : ( i1, si32, si32  ) -> ()

    // CHECK: omp.barrier
    omp.barrier

    return
}

// CHECK: omp.reduction.declare
// CHECK-LABEL: @add_f32
// CHECK: : f32
// CHECK: init
// CHECK: ^{{.+}}(%{{.+}}: f32):
// CHECK:   omp.yield
// CHECK: combiner
// CHECK: ^{{.+}}(%{{.+}}: f32, %{{.+}}: f32):
// CHECK:   omp.yield
// CHECK: atomic
// CHECK: ^{{.+}}(%{{.+}}: !llvm.ptr<f32>, %{{.+}}: !llvm.ptr<f32>):
// CHECK:  omp.yield
omp.reduction.declare @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
atomic {
^bb2(%arg2: !llvm.ptr<f32>, %arg3: !llvm.ptr<f32>):
  %2 = llvm.load %arg3 : !llvm.ptr<f32>
  llvm.atomicrmw fadd %arg2, %2 monotonic : f32
  omp.yield
}

func @reduction(%lb : index, %ub : index, %step : index) {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<f32>
  // CHECK: reduction(@add_f32 -> %{{.+}} : !llvm.ptr<f32>)
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step)
  reduction(@add_f32 -> %0 : !llvm.ptr<f32>) {
    %1 = arith.constant 2.0 : f32
    // CHECK: omp.reduction %{{.+}}, %{{.+}}
    omp.reduction %1, %0 : !llvm.ptr<f32>
    omp.yield
  }
  return
}

// CHECK: omp.reduction.declare
// CHECK-LABEL: @add2_f32
omp.reduction.declare @add2_f32 : f32
// CHECK: init
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
// CHECK: combiner
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
// CHECK-NOT: atomic

func @reduction2(%lb : index, %ub : index, %step : index) {
  %0 = memref.alloca() : memref<1xf32>
  // CHECK: reduction
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step)
  reduction(@add2_f32 -> %0 : memref<1xf32>) {
    %1 = arith.constant 2.0 : f32
    // CHECK: omp.reduction
    omp.reduction %1, %0 : memref<1xf32>
    omp.yield
  }
  return
}

// CHECK: omp.critical.declare @mutex1 hint(uncontended)
omp.critical.declare @mutex1 hint(uncontended)
// CHECK: omp.critical.declare @mutex2 hint(contended)
omp.critical.declare @mutex2 hint(contended)
// CHECK: omp.critical.declare @mutex3 hint(nonspeculative)
omp.critical.declare @mutex3 hint(nonspeculative)
// CHECK: omp.critical.declare @mutex4 hint(speculative)
omp.critical.declare @mutex4 hint(speculative)
// CHECK: omp.critical.declare @mutex5 hint(uncontended, nonspeculative)
omp.critical.declare @mutex5 hint(uncontended, nonspeculative)
// CHECK: omp.critical.declare @mutex6 hint(contended, nonspeculative)
omp.critical.declare @mutex6 hint(contended, nonspeculative)
// CHECK: omp.critical.declare @mutex7 hint(uncontended, speculative)
omp.critical.declare @mutex7 hint(uncontended, speculative)
// CHECK: omp.critical.declare @mutex8 hint(contended, speculative)
omp.critical.declare @mutex8 hint(contended, speculative)


// CHECK-LABEL: omp_critical
func @omp_critical() -> () {
  // CHECK: omp.critical
  omp.critical {
    omp.terminator
  }

  // CHECK: omp.critical(@{{.*}})
  omp.critical(@mutex1) {
    omp.terminator
  }
  return
}

func @omp_ordered(%arg1 : i32, %arg2 : i32, %arg3 : i32,
    %vec0 : i64, %vec1 : i64, %vec2 : i64, %vec3 : i64) -> () {
  // CHECK: omp.ordered_region
  omp.ordered_region {
    // CHECK: omp.terminator
    omp.terminator
  }

  omp.wsloop (%0) : i32 = (%arg1) to (%arg2) step (%arg3) ordered(0) {
    omp.ordered_region {
      omp.terminator
    }
    omp.yield
  }

  omp.wsloop (%0) : i32 = (%arg1) to (%arg2) step (%arg3) ordered(1) {
    // Only one DEPEND(SINK: vec) clause
    // CHECK: omp.ordered depend_type("dependsink") depend_vec(%{{.*}} : i64) {num_loops_val = 1 : i64}
    omp.ordered depend_type("dependsink") depend_vec(%vec0 : i64) {num_loops_val = 1 : i64}

    // CHECK: omp.ordered depend_type("dependsource") depend_vec(%{{.*}} : i64) {num_loops_val = 1 : i64}
    omp.ordered depend_type("dependsource") depend_vec(%vec0 : i64) {num_loops_val = 1 : i64}

    omp.yield
  }

  omp.wsloop (%0) : i32 = (%arg1) to (%arg2) step (%arg3) ordered(2) {
    // Multiple DEPEND(SINK: vec) clauses
    // CHECK: omp.ordered depend_type("dependsink") depend_vec(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i64, i64, i64, i64) {num_loops_val = 2 : i64}
    omp.ordered depend_type("dependsink") depend_vec(%vec0, %vec1, %vec2, %vec3 : i64, i64, i64, i64) {num_loops_val = 2 : i64}

    // CHECK: omp.ordered depend_type("dependsource") depend_vec(%{{.*}}, %{{.*}} : i64, i64) {num_loops_val = 2 : i64}
    omp.ordered depend_type("dependsource") depend_vec(%vec0, %vec1 : i64, i64) {num_loops_val = 2 : i64}

    omp.yield
  }

  return
}

// CHECK-LABEL: omp_atomic_read
// CHECK-SAME: (%[[v:.*]]: memref<i32>, %[[x:.*]]: memref<i32>)
func @omp_atomic_read(%v: memref<i32>, %x: memref<i32>) {
  // CHECK: omp.atomic.read %[[v]] = %[[x]] : memref<i32>
  omp.atomic.read %v = %x : memref<i32>
  // CHECK: omp.atomic.read %[[v]] = %[[x]] memory_order(seq_cst) : memref<i32>
  omp.atomic.read %v = %x memory_order(seq_cst) : memref<i32>
  // CHECK: omp.atomic.read %[[v]] = %[[x]] memory_order(acquire) : memref<i32>
  omp.atomic.read %v = %x memory_order(acquire) : memref<i32>
  // CHECK: omp.atomic.read %[[v]] = %[[x]] memory_order(relaxed) : memref<i32>
  omp.atomic.read %v = %x memory_order(relaxed) : memref<i32>
  // CHECK: omp.atomic.read %[[v]] = %[[x]] hint(contended, nonspeculative) : memref<i32>
  omp.atomic.read %v = %x hint(nonspeculative, contended) : memref<i32>
  // CHECK: omp.atomic.read %[[v]] = %[[x]] memory_order(seq_cst) hint(contended, speculative) : memref<i32>
  omp.atomic.read %v = %x hint(speculative, contended) memory_order(seq_cst) : memref<i32>
  return
}

// CHECK-LABEL: omp_atomic_write
// CHECK-SAME: (%[[ADDR:.*]]: memref<i32>, %[[VAL:.*]]: i32)
func @omp_atomic_write(%addr : memref<i32>, %val : i32) {
  // CHECK: omp.atomic.write %[[ADDR]] = %[[VAL]] : memref<i32>, i32
  omp.atomic.write %addr = %val : memref<i32>, i32
  // CHECK: omp.atomic.write %[[ADDR]] = %[[VAL]] memory_order(seq_cst) : memref<i32>, i32
  omp.atomic.write %addr = %val memory_order(seq_cst) : memref<i32>, i32
  // CHECK: omp.atomic.write %[[ADDR]] = %[[VAL]] memory_order(release) : memref<i32>, i32
  omp.atomic.write %addr = %val memory_order(release) : memref<i32>, i32
  // CHECK: omp.atomic.write %[[ADDR]] = %[[VAL]] memory_order(relaxed) : memref<i32>, i32
  omp.atomic.write %addr = %val memory_order(relaxed) : memref<i32>, i32
  // CHECK: omp.atomic.write %[[ADDR]] = %[[VAL]] hint(uncontended, speculative) : memref<i32>, i32
  omp.atomic.write %addr = %val hint(speculative, uncontended) : memref<i32>, i32
  return
}

// CHECK-LABEL: omp_atomic_update
// CHECK-SAME: (%[[X:.*]]: memref<i32>, %[[EXPR:.*]]: i32, %[[XBOOL:.*]]: memref<i1>, %[[EXPRBOOL:.*]]: i1)
func @omp_atomic_update(%x : memref<i32>, %expr : i32, %xBool : memref<i1>, %exprBool : i1) {
  // CHECK: omp.atomic.update %[[X]] = %[[X]] add %[[EXPR]] : memref<i32>, i32
  omp.atomic.update %x = %x add %expr : memref<i32>, i32
  // CHECK: omp.atomic.update %[[X]] = %[[X]] sub %[[EXPR]] : memref<i32>, i32
  omp.atomic.update %x = %x sub %expr : memref<i32>, i32
  // CHECK: omp.atomic.update %[[X]] = %[[X]] mul %[[EXPR]] : memref<i32>, i32
  omp.atomic.update %x = %x mul %expr : memref<i32>, i32
  // CHECK: omp.atomic.update %[[X]] = %[[X]] div %[[EXPR]] : memref<i32>, i32
  omp.atomic.update %x = %x div %expr : memref<i32>, i32
  // CHECK: omp.atomic.update %[[XBOOL]] = %[[XBOOL]] and %[[EXPRBOOL]] : memref<i1>, i1
  omp.atomic.update %xBool = %xBool and %exprBool : memref<i1>, i1
  // CHECK: omp.atomic.update %[[XBOOL]] = %[[XBOOL]] or %[[EXPRBOOL]] : memref<i1>, i1
  omp.atomic.update %xBool = %xBool or %exprBool : memref<i1>, i1
  // CHECK: omp.atomic.update %[[XBOOL]] = %[[XBOOL]] xor %[[EXPRBOOL]] : memref<i1>, i1
  omp.atomic.update %xBool = %xBool xor %exprBool : memref<i1>, i1
  // CHECK: omp.atomic.update %[[X]] = %[[X]] shiftr %[[EXPR]] : memref<i32>, i32
  omp.atomic.update %x = %x shiftr %expr : memref<i32>, i32
  // CHECK: omp.atomic.update %[[X]] = %[[X]] shiftl %[[EXPR]] : memref<i32>, i32
  omp.atomic.update %x = %x shiftl %expr : memref<i32>, i32
  // CHECK: omp.atomic.update %[[X]] = %[[X]] max %[[EXPR]] : memref<i32>, i32
  omp.atomic.update %x = %x max %expr : memref<i32>, i32
  // CHECK: omp.atomic.update %[[X]] = %[[X]] min %[[EXPR]] : memref<i32>, i32
  omp.atomic.update %x = %x min %expr : memref<i32>, i32
  // CHECK: omp.atomic.update %[[XBOOL]] = %[[XBOOL]] eqv %[[EXPRBOOL]] : memref<i1>, i1
  omp.atomic.update %xBool = %xBool eqv %exprBool : memref<i1>, i1
  // CHECK: omp.atomic.update %[[XBOOL]] = %[[XBOOL]] neqv %[[EXPRBOOL]] : memref<i1>, i1
  omp.atomic.update %xBool = %xBool neqv %exprBool : memref<i1>, i1

  // CHECK: omp.atomic.update %[[X]] = %[[EXPR]] add %[[X]] : memref<i32>, i32
  omp.atomic.update %x = %expr add %x : memref<i32>, i32
  // CHECK: omp.atomic.update %[[X]] = %[[EXPR]] sub %[[X]] : memref<i32>, i32
  omp.atomic.update %x = %expr sub %x : memref<i32>, i32
  // CHECK: omp.atomic.update %[[X]] = %[[EXPR]] mul %[[X]] : memref<i32>, i32
  omp.atomic.update %x = %expr mul %x : memref<i32>, i32
  // CHECK: omp.atomic.update %[[X]] = %[[EXPR]] div %[[X]] : memref<i32>, i32
  omp.atomic.update %x = %expr div %x : memref<i32>, i32
  // CHECK: omp.atomic.update %[[XBOOL]] = %[[EXPRBOOL]] and %[[XBOOL]] : memref<i1>, i1
  omp.atomic.update %xBool =  %exprBool and %xBool : memref<i1>, i1
  // CHECK: omp.atomic.update %[[XBOOL]] = %[[EXPRBOOL]] or %[[XBOOL]] : memref<i1>, i1
  omp.atomic.update %xBool =  %exprBool or %xBool : memref<i1>, i1
  // CHECK: omp.atomic.update %[[XBOOL]] = %[[EXPRBOOL]] xor %[[XBOOL]] : memref<i1>, i1
  omp.atomic.update %xBool =  %exprBool xor %xBool : memref<i1>, i1
  // CHECK: omp.atomic.update %[[X]] = %[[EXPR]] shiftr %[[X]] : memref<i32>, i32
  omp.atomic.update %x =  %expr shiftr %x : memref<i32>, i32
  // CHECK: omp.atomic.update %[[X]] = %[[EXPR]] shiftl %[[X]] : memref<i32>, i32
  omp.atomic.update %x = %expr shiftl %x : memref<i32>, i32
  // CHECK: omp.atomic.update %[[X]] = %[[EXPR]] max %[[X]] : memref<i32>, i32
  omp.atomic.update %x = %expr max %x : memref<i32>, i32
  // CHECK: omp.atomic.update %[[X]] = %[[EXPR]] min %[[X]] : memref<i32>, i32
  omp.atomic.update %x = %expr min %x : memref<i32>, i32
  // CHECK: omp.atomic.update %[[XBOOL]] = %[[EXPRBOOL]] eqv %[[XBOOL]] : memref<i1>, i1
  omp.atomic.update %xBool =  %exprBool eqv %xBool : memref<i1>, i1
  // CHECK: omp.atomic.update %[[XBOOL]] = %[[EXPRBOOL]] neqv %[[XBOOL]] : memref<i1>, i1
  omp.atomic.update %xBool =  %exprBool neqv %xBool : memref<i1>, i1
  // CHECK: omp.atomic.update %[[X]] = %[[EXPR]] add %[[X]] memory_order(seq_cst) hint(speculative) : memref<i32>, i32
  omp.atomic.update %x = %expr add %x hint(speculative) memory_order(seq_cst) : memref<i32>, i32
  return
}

// CHECK-LABEL: omp_sectionsop
func @omp_sectionsop(%data_var1 : memref<i32>, %data_var2 : memref<i32>,
                     %data_var3 : memref<i32>, %redn_var : !llvm.ptr<f32>) {

  // CHECK: omp.sections private(%{{.*}} : memref<i32>) {
  "omp.sections" (%data_var1) ({
    // CHECK: omp.terminator
    omp.terminator
  }) {operand_segment_sizes = dense<[1,0,0,0,0,0]> : vector<6xi32>} : (memref<i32>) -> ()

  // CHECK: omp.sections firstprivate(%{{.*}} : memref<i32>) {
  "omp.sections" (%data_var1) ({
    // CHECK: omp.terminator
    omp.terminator
  }) {operand_segment_sizes = dense<[0,1,0,0,0,0]> : vector<6xi32>} : (memref<i32>) -> ()

  // CHECK: omp.sections lastprivate(%{{.*}} : memref<i32>) {
  "omp.sections" (%data_var1) ({
    // CHECK: omp.terminator
    omp.terminator
  }) {operand_segment_sizes = dense<[0,0,1,0,0,0]> : vector<6xi32>} : (memref<i32>) -> ()

  // CHECK: omp.sections private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) lastprivate(%{{.*}} : memref<i32>) {
  "omp.sections" (%data_var1, %data_var2, %data_var3) ({
    // CHECK: omp.terminator
    omp.terminator
  }) {operand_segment_sizes = dense<[1,1,1,0,0,0]> : vector<6xi32>} : (memref<i32>, memref<i32>, memref<i32>) -> ()

  // CHECK: omp.sections allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
  "omp.sections" (%data_var1, %data_var1) ({
    // CHECK: omp.terminator
    omp.terminator
  }) {operand_segment_sizes = dense<[0,0,0,0,1,1]> : vector<6xi32>} : (memref<i32>, memref<i32>) -> ()

    // CHECK: omp.sections reduction(@add_f32 -> %{{.*}} : !llvm.ptr<f32>)
  "omp.sections" (%redn_var) ({
    // CHECK: omp.terminator
    omp.terminator
  }) {operand_segment_sizes = dense<[0,0,0,1,0,0]> : vector<6xi32>, reductions=[@add_f32]} : (!llvm.ptr<f32>) -> ()

  // CHECK: omp.sections private(%{{.*}} : memref<i32>) {
  omp.sections private(%data_var1 : memref<i32>) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.sections firstprivate(%{{.*}} : memref<i32>)
  omp.sections firstprivate(%data_var1 : memref<i32>) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.sections lastprivate(%{{.*}} : memref<i32>)
  omp.sections lastprivate(%data_var1 : memref<i32>) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.sections private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) lastprivate(%{{.*}} : memref<i32>) {
  omp.sections private(%data_var1 : memref<i32>) firstprivate(%data_var2 : memref<i32>) lastprivate(%data_var3 : memref<i32>) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.sections private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) lastprivate(%{{.*}} : memref<i32>) {
  omp.sections lastprivate(%data_var1 : memref<i32>) firstprivate(%data_var2 : memref<i32>) private(%data_var3 : memref<i32>) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.sections private(%{{.*}} : memref<i32>) nowait {
  omp.sections nowait private(%data_var1 : memref<i32>) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.sections firstprivate(%{{.*}} : memref<i32>, %{{.*}} : memref<i32>) lastprivate(%{{.*}} : memref<i32>) {
  omp.sections firstprivate(%data_var1 : memref<i32>, %data_var2 : memref<i32>) lastprivate(%data_var1 : memref<i32>) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.sections reduction(@add_f32 -> %{{.*}} : !llvm.ptr<f32>) {
  omp.sections reduction(@add_f32 -> %redn_var : !llvm.ptr<f32>) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.sections allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
  omp.sections allocate(%data_var1 : memref<i32> -> %data_var1 : memref<i32>) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.sections nowait
  omp.sections nowait {
    // CHECK: omp.section
    omp.section {
      // CHECK: %{{.*}} = "test.payload"() : () -> i32
      %1 = "test.payload"() : () -> i32
      // CHECK: %{{.*}} = "test.payload"() : () -> i32
      %2 = "test.payload"() : () -> i32
      // CHECK: %{{.*}} = "test.payload"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
      %3 = "test.payload"(%1, %2) : (i32, i32) -> i32
    }
    // CHECK: omp.section
    omp.section {
      // CHECK: %{{.*}} = "test.payload"(%{{.*}}) : (!llvm.ptr<f32>) -> i32
      %1 = "test.payload"(%redn_var) : (!llvm.ptr<f32>) -> i32
    }
    // CHECK: omp.section
    omp.section {
      // CHECK: "test.payload"(%{{.*}}) : (!llvm.ptr<f32>) -> ()
      "test.payload"(%redn_var) : (!llvm.ptr<f32>) -> ()
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}
