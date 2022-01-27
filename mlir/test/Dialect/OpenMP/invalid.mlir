// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func @unknown_clause() {
  // expected-error@+1 {{invalid is not a valid clause}}
  omp.parallel invalid {
  }

  return
}

// -----

func @if_once(%n : i1) {
  // expected-error@+1 {{at most one if clause can appear on the omp.parallel operation}}
  omp.parallel if(%n : i1) if(%n : i1) {
  }

  return
}

// -----

func @num_threads_once(%n : si32) {
  // expected-error@+1 {{at most one num_threads clause can appear on the omp.parallel operation}}
  omp.parallel num_threads(%n : si32) num_threads(%n : si32) {
  }

  return
}

// -----

func @private_once(%n : memref<i32>) {
  // expected-error@+1 {{at most one private clause can appear on the omp.parallel operation}}
  omp.parallel private(%n : memref<i32>) private(%n : memref<i32>) {
  }

  return
}

// -----

func @firstprivate_once(%n : memref<i32>) {
  // expected-error@+1 {{at most one firstprivate clause can appear on the omp.parallel operation}}
  omp.parallel firstprivate(%n : memref<i32>) firstprivate(%n : memref<i32>) {
  }

  return
}

// -----

func @shared_once(%n : memref<i32>) {
  // expected-error@+1 {{at most one shared clause can appear on the omp.parallel operation}}
  omp.parallel shared(%n : memref<i32>) shared(%n : memref<i32>) {
  }

  return
}

// -----

func @copyin_once(%n : memref<i32>) {
  // expected-error@+1 {{at most one copyin clause can appear on the omp.parallel operation}}
  omp.parallel copyin(%n : memref<i32>) copyin(%n : memref<i32>) {
  }

  return
}

// -----

func @lastprivate_not_allowed(%n : memref<i32>) {
  // expected-error@+1 {{lastprivate is not a valid clause for the omp.parallel operation}}
  omp.parallel lastprivate(%n : memref<i32>) {}
  return
}

// -----

func @nowait_not_allowed(%n : memref<i32>) {
  // expected-error@+1 {{nowait is not a valid clause for the omp.parallel operation}}
  omp.parallel nowait {}
  return
}

// -----

func @linear_not_allowed(%data_var : memref<i32>, %linear_var : i32) {
  // expected-error@+1 {{linear is not a valid clause for the omp.parallel operation}}
  omp.parallel linear(%data_var = %linear_var : memref<i32>)  {}
  return
}

// -----

func @schedule_not_allowed() {
  // expected-error@+1 {{schedule is not a valid clause for the omp.parallel operation}}
  omp.parallel schedule(static) {}
  return
}

// -----

func @collapse_not_allowed() {
  // expected-error@+1 {{collapse is not a valid clause for the omp.parallel operation}}
  omp.parallel collapse(3) {}
  return
}

// -----

func @order_not_allowed() {
  // expected-error@+1 {{order is not a valid clause for the omp.parallel operation}}
  omp.parallel order(concurrent) {}
  return
}

// -----

func @ordered_not_allowed() {
  // expected-error@+1 {{ordered is not a valid clause for the omp.parallel operation}}
  omp.parallel ordered(2) {}
}

// -----

func @default_once() {
  // expected-error@+1 {{at most one default clause can appear on the omp.parallel operation}}
  omp.parallel default(private) default(firstprivate) {
  }

  return
}

// -----

func @proc_bind_once() {
  // expected-error@+1 {{at most one proc_bind clause can appear on the omp.parallel operation}}
  omp.parallel proc_bind(close) proc_bind(spread) {
  }

  return
}

// -----

func @inclusive_not_a_clause(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{inclusive is not a valid clause}}
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) nowait inclusive {
    omp.yield
  }
}

// -----

func @order_value(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{invalid order kind}}
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) order(default) {
    omp.yield
  }
}

// -----

func @shared_not_allowed(%lb : index, %ub : index, %step : index, %var : memref<i32>) {
  // expected-error @below {{shared is not a valid clause for the omp.wsloop operation}}
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) shared(%var) {
    omp.yield
  }
}

// -----

func @copyin(%lb : index, %ub : index, %step : index, %var : memref<i32>) {
  // expected-error @below {{copyin is not a valid clause for the omp.wsloop operation}}
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) copyin(%var) {
    omp.yield
  }
}

// -----

func @if_not_allowed(%lb : index, %ub : index, %step : index, %bool_var : i1) {
  // expected-error @below {{if is not a valid clause for the omp.wsloop operation}}
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) if(%bool_var: i1) {
    omp.yield
  }
}

// -----

func @num_threads_not_allowed(%lb : index, %ub : index, %step : index, %int_var : i32) {
  // expected-error @below {{num_threads is not a valid clause for the omp.wsloop operation}}
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) num_threads(%int_var: i32) {
    omp.yield
  }
}

// -----

func @default_not_allowed(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{default is not a valid clause for the omp.wsloop operation}}
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) default(private) {
    omp.yield
  }
}

// -----

func @proc_bind_not_allowed(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{proc_bind is not a valid clause for the omp.wsloop operation}}
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) proc_bind(close) {
    omp.yield
  }
}

// -----

// expected-error @below {{op expects initializer region with one argument of the reduction type}}
omp.reduction.declare @add_f32 : f64
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

// -----

// expected-error @below {{expects initializer region to yield a value of the reduction type}}
omp.reduction.declare @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f64
  omp.yield (%0 : f64)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}

// -----

// expected-error @below {{expects reduction region with two arguments of the reduction type}}
omp.reduction.declare @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f64, %arg1: f64):
  %1 = arith.addf %arg0, %arg1 : f64
  omp.yield (%1 : f64)
}

// -----

// expected-error @below {{expects reduction region to yield a value of the reduction type}}
omp.reduction.declare @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  %2 = arith.extf %1 : f32 to f64
  omp.yield (%2 : f64)
}

// -----

// expected-error @below {{expects atomic reduction region with two arguments of the same type}}
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
^bb2(%arg0: memref<f32>, %arg1: memref<f64>):
  omp.yield
}

// -----

// expected-error @below {{expects atomic reduction region arguments to be accumulators containing the reduction type}}
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
^bb2(%arg0: memref<f64>, %arg1: memref<f64>):
  omp.yield
}

// -----

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

func @foo(%lb : index, %ub : index, %step : index) {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<f32>
  %1 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<f32>

  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step)
  reduction(@add_f32 -> %0 : !llvm.ptr<f32>) {
    %2 = arith.constant 2.0 : f32
    // expected-error @below {{accumulator is not used by the parent}}
    omp.reduction %2, %1 : !llvm.ptr<f32>
    omp.yield
  }
  return
}

// -----

func @foo(%lb : index, %ub : index, %step : index) {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<f32>
  %1 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<f32>

  // expected-error @below {{expected symbol reference @foo to point to a reduction declaration}}
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step)
  reduction(@foo -> %0 : !llvm.ptr<f32>) {
    %2 = arith.constant 2.0 : f32
    omp.reduction %2, %1 : !llvm.ptr<f32>
    omp.yield
  }
  return
}

// -----

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

func @foo(%lb : index, %ub : index, %step : index) {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<f32>

  // expected-error @below {{accumulator variable used more than once}}
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step)
  reduction(@add_f32 -> %0 : !llvm.ptr<f32>, @add_f32 -> %0 : !llvm.ptr<f32>) {
    %2 = arith.constant 2.0 : f32
    omp.reduction %2, %0 : !llvm.ptr<f32>
    omp.yield
  }
  return
}

// -----

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

func @foo(%lb : index, %ub : index, %step : index, %mem : memref<1xf32>) {
  %c1 = arith.constant 1 : i32

  // expected-error @below {{expected accumulator ('memref<1xf32>') to be the same type as reduction declaration ('!llvm.ptr<f32>')}}
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step)
  reduction(@add_f32 -> %mem : memref<1xf32>) {
    %2 = arith.constant 2.0 : f32
    omp.reduction %2, %mem : memref<1xf32>
    omp.yield
  }
  return
}

// -----

func @omp_critical2() -> () {
  // expected-error @below {{expected symbol reference @excl to point to a critical declaration}}
  omp.critical(@excl) {
    omp.terminator
  }
  return
}

// -----

// expected-error @below {{the hints omp_sync_hint_uncontended and omp_sync_hint_contended cannot be combined}}
omp.critical.declare @mutex hint(uncontended, contended)

// -----

// expected-error @below {{the hints omp_sync_hint_nonspeculative and omp_sync_hint_speculative cannot be combined}}
omp.critical.declare @mutex hint(nonspeculative, speculative)

// -----

// expected-error @below {{invalid_hint is not a valid hint}}
omp.critical.declare @mutex hint(invalid_hint)

// -----

func @omp_ordered1(%arg1 : i32, %arg2 : i32, %arg3 : i32) -> () {
  omp.wsloop (%0) : i32 = (%arg1) to (%arg2) step (%arg3) ordered(1) {
    // expected-error @below {{ordered region must be closely nested inside a worksharing-loop region with an ordered clause without parameter present}}
    omp.ordered_region {
      omp.terminator
    }
    omp.yield
  }
  return
}

// -----

func @omp_ordered2(%arg1 : i32, %arg2 : i32, %arg3 : i32) -> () {
  omp.wsloop (%0) : i32 = (%arg1) to (%arg2) step (%arg3) {
    // expected-error @below {{ordered region must be closely nested inside a worksharing-loop region with an ordered clause without parameter present}}
    omp.ordered_region {
      omp.terminator
    }
    omp.yield
  }
  return
}

// -----

func @omp_ordered3(%vec0 : i64) -> () {
  // expected-error @below {{ordered depend directive must be closely nested inside a worksharing-loop with ordered clause with parameter present}}
  omp.ordered depend_type(dependsink) depend_vec(%vec0 : i64) {num_loops_val = 1 : i64}
  return
}

// -----

func @omp_ordered4(%arg1 : i32, %arg2 : i32, %arg3 : i32, %vec0 : i64) -> () {
  omp.wsloop (%0) : i32 = (%arg1) to (%arg2) step (%arg3) ordered(0) {
    // expected-error @below {{ordered depend directive must be closely nested inside a worksharing-loop with ordered clause with parameter present}}
    omp.ordered depend_type(dependsink) depend_vec(%vec0 : i64) {num_loops_val = 1 : i64}

    omp.yield
  }
  return
}
// -----

func @omp_ordered5(%arg1 : i32, %arg2 : i32, %arg3 : i32, %vec0 : i64, %vec1 : i64) -> () {
  omp.wsloop (%0) : i32 = (%arg1) to (%arg2) step (%arg3) ordered(1) {
    // expected-error @below {{number of variables in depend clause does not match number of iteration variables in the doacross loop}}
    omp.ordered depend_type(dependsource) depend_vec(%vec0, %vec1 : i64, i64) {num_loops_val = 2 : i64}

    omp.yield
  }
  return
}

// -----

func @omp_atomic_read1(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{the hints omp_sync_hint_nonspeculative and omp_sync_hint_speculative cannot be combined.}}
  omp.atomic.read %v = %x hint(speculative, nonspeculative) : memref<i32>
  return
}

// -----

func @omp_atomic_read2(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{invalid memory order kind}}
  omp.atomic.read %v = %x memory_order(xyz) : memref<i32>
  return
}

// -----

func @omp_atomic_read3(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{memory-order must not be acq_rel or release for atomic reads}}
  omp.atomic.read %v = %x memory_order(acq_rel) : memref<i32>
  return
}

// -----

func @omp_atomic_read4(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{memory-order must not be acq_rel or release for atomic reads}}
  omp.atomic.read %v = %x memory_order(release) : memref<i32>
  return
}

// -----

func @omp_atomic_read5(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{at most one memory_order clause can appear on the omp.atomic.read operation}}
  omp.atomic.read %v = %x memory_order(acquire) memory_order(relaxed) : memref<i32>
  return
}

// -----

func @omp_atomic_read6(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{at most one hint clause can appear on the omp.atomic.read operation}}
  omp.atomic.read %v =  %x hint(speculative) hint(contended) : memref<i32>
  return
}

// -----

func @omp_atomic_read6(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{read and write must not be to the same location for atomic reads}}
  omp.atomic.read %x =  %x hint(speculative) : memref<i32>
  return
}

// -----

func @omp_atomic_write1(%addr : memref<i32>, %val : i32) {
  // expected-error @below {{the hints omp_sync_hint_uncontended and omp_sync_hint_contended cannot be combined}}
  omp.atomic.write  %addr = %val hint(contended, uncontended) : memref<i32>, i32
  return
}

// -----

func @omp_atomic_write2(%addr : memref<i32>, %val : i32) {
  // expected-error @below {{memory-order must not be acq_rel or acquire for atomic writes}}
  omp.atomic.write  %addr = %val memory_order(acq_rel) : memref<i32>, i32
  return
}

// -----

func @omp_atomic_write3(%addr : memref<i32>, %val : i32) {
  // expected-error @below {{memory-order must not be acq_rel or acquire for atomic writes}}
  omp.atomic.write  %addr = %val memory_order(acquire) : memref<i32>, i32
  return
}

// -----

func @omp_atomic_write4(%addr : memref<i32>, %val : i32) {
  // expected-error @below {{at most one memory_order clause can appear on the omp.atomic.write operation}}
  omp.atomic.write  %addr = %val memory_order(release) memory_order(seq_cst) : memref<i32>, i32
  return
}

// -----

func @omp_atomic_write5(%addr : memref<i32>, %val : i32) {
  // expected-error @below {{at most one hint clause can appear on the omp.atomic.write operation}}
  omp.atomic.write  %addr = %val hint(contended) hint(speculative) : memref<i32>, i32
  return
}

// -----

func @omp_atomic_write6(%addr : memref<i32>, %val : i32) {
  // expected-error @below {{invalid memory order kind}}
  omp.atomic.write  %addr = %val memory_order(xyz) : memref<i32>, i32
  return
}

// -----

func @omp_atomic_update1(%x: memref<i32>, %expr: i32, %foo: memref<i32>) {
  // expected-error @below {{atomic update variable %x not found in the RHS of the assignment statement in an atomic.update operation}}
  omp.atomic.update %x = %foo add %expr : memref<i32>, i32
  return
}

// -----

func @omp_atomic_update2(%x: memref<i32>, %expr: i32) {
  // expected-error @below {{invalid atomic bin op in atomic update}}
  omp.atomic.update %x = %x invalid %expr : memref<i32>, i32
  return
}

// -----

func @omp_atomic_update3(%x: memref<i32>, %expr: i32) {
  // expected-error @below {{memory-order must not be acq_rel or acquire for atomic updates}}
  omp.atomic.update %x = %x add %expr memory_order(acq_rel) : memref<i32>, i32
  return
}

// -----

func @omp_atomic_update4(%x: memref<i32>, %expr: i32) {
  // expected-error @below {{memory-order must not be acq_rel or acquire for atomic updates}}
  omp.atomic.update %x = %x add %expr memory_order(acquire) : memref<i32>, i32
  return
}

// -----

// expected-note @below {{prior use here}}
func @omp_atomic_update5(%x: memref<i32>, %expr: i32) {
  // expected-error @below {{use of value '%x' expects different type than prior uses: 'i32' vs 'memref<i32>'}}
  omp.atomic.update %x = %x add %expr : i32, memref<i32>
  return
}

// -----

func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  // expected-error @below {{expected three operations in omp.atomic.capture region}}
  omp.atomic.capture {
    omp.atomic.read %v = %x : memref<i32>
    omp.terminator
  }
  return
}

// -----

func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    omp.atomic.read %v = %x : memref<i32>
    omp.atomic.read %v = %x : memref<i32>
    omp.terminator
  }
  return
}

// -----

func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    omp.atomic.update %x = %x add %expr : memref<i32>, i32
    omp.atomic.update %x = %x sub %expr : memref<i32>, i32
    omp.terminator
  }
  return
}

// -----

func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    omp.atomic.write %x = %expr : memref<i32>, i32
    omp.atomic.write %x = %expr : memref<i32>, i32
    omp.terminator
  }
  return
}

// -----

func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    omp.atomic.write %x = %expr : memref<i32>, i32
    omp.atomic.update %x = %x add %expr : memref<i32>, i32
    omp.terminator
  }
  return
}

// -----

func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    omp.atomic.update %x = %x add %expr : memref<i32>, i32
    omp.atomic.write %x = %expr : memref<i32>, i32
    omp.terminator
  }
  return
}

// -----

func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    omp.atomic.write %x = %expr : memref<i32>, i32
    omp.atomic.read %v = %x : memref<i32>
    omp.terminator
  }
  return
}

// -----

func @omp_atomic_capture(%x: memref<i32>, %y: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{updated variable in omp.atomic.update must be captured in second operation}}
    omp.atomic.update %x = %x add %expr : memref<i32>, i32
    omp.atomic.read %v = %y : memref<i32>
    omp.terminator
  }
}

// -----

func @omp_atomic_capture(%x: memref<i32>, %y: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{captured variable in omp.atomic.read must be updated in second operation}}
    omp.atomic.read %v = %y : memref<i32>
    omp.atomic.update %x = %x add %expr : memref<i32>, i32
    omp.terminator
  }
}

// -----

func @omp_atomic_capture(%x: memref<i32>, %y: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{captured variable in omp.atomic.read must be updated in second operation}}
    omp.atomic.read %v = %x : memref<i32>
    omp.atomic.write %y = %expr : memref<i32>, i32
    omp.terminator
  }
}

// -----

func @omp_sections(%data_var1 : memref<i32>, %data_var2 : memref<i32>, %data_var3 : memref<i32>) -> () {
  // expected-error @below {{operand used in both private and firstprivate clauses}}
  omp.sections private(%data_var1 : memref<i32>) firstprivate(%data_var1 : memref<i32>) {
    omp.terminator
  }
  return
}

// -----

func @omp_sections(%data_var1 : memref<i32>, %data_var2 : memref<i32>, %data_var3 : memref<i32>) -> () {
  // expected-error @below {{operand used in both private and lastprivate clauses}}
  omp.sections private(%data_var1 : memref<i32>) lastprivate(%data_var1 : memref<i32>) {
    omp.terminator
  }
  return
}

// -----

func @omp_sections(%data_var1 : memref<i32>, %data_var2 : memref<i32>, %data_var3 : memref<i32>) -> () {
  // expected-error @below {{operand used in both private and lastprivate clauses}}
  omp.sections private(%data_var1 : memref<i32>, %data_var2 : memref<i32>) lastprivate(%data_var3 : memref<i32>, %data_var2 : memref<i32>) {
    omp.terminator
  }
  return
}

// -----

func @omp_sections(%data_var : memref<i32>) -> () {
  // expected-error @below {{expected equal sizes for allocate and allocator variables}}
  "omp.sections" (%data_var) ({
    omp.terminator
  }) {operand_segment_sizes = dense<[0,0,0,0,1,0]> : vector<6xi32>} : (memref<i32>) -> ()
  return
}

// -----

func @omp_sections(%data_var : memref<i32>) -> () {
  // expected-error @below {{expected as many reduction symbol references as reduction variables}}
  "omp.sections" (%data_var) ({
    omp.terminator
  }) {operand_segment_sizes = dense<[0,0,0,1,0,0]> : vector<6xi32>} : (memref<i32>) -> ()
  return
}

// -----

func @omp_sections(%data_var : memref<i32>) -> () {
  // expected-error @below {{expected omp.section op or terminator op inside region}}
  omp.sections {
    "test.payload" () : () -> ()
  }
  return
}

// -----

func @omp_sections(%cond : i1) {
  // expected-error @below {{if is not a valid clause for the omp.sections operation}}
  omp.sections if(%cond) {
    omp.terminator
  }
  return
}

// -----

func @omp_sections() {
  // expected-error @below {{num_threads is not a valid clause for the omp.sections operation}}
  omp.sections num_threads(10) {
    omp.terminator
  }
  return
}

// -----

func @omp_sections(%datavar : memref<i32>) {
  // expected-error @below {{shared is not a valid clause for the omp.sections operation}}
  omp.sections shared(%datavar : memref<i32>) {
    omp.terminator
  }
  return
}

// -----

func @omp_sections(%datavar : memref<i32>) {
  // expected-error @below {{copyin is not a valid clause for the omp.sections operation}}
  omp.sections copyin(%datavar : memref<i32>) {
    omp.terminator
  }
  return
}

// -----

func @omp_sections() {
  // expected-error @below {{default is not a valid clause for the omp.sections operation}}
  omp.sections default(private) {
    omp.terminator
  }
  return
}

// -----

func @omp_sections() {
  // expected-error @below {{proc_bind is not a valid clause for the omp.sections operation}}
  omp.sections proc_bind(close) {
    omp.terminator
  }
  return
}

// -----

func @omp_sections(%data_var : memref<i32>, %linear_var : i32) {
  // expected-error @below {{linear is not a valid clause for the omp.sections operation}}
  omp.sections linear(%data_var = %linear_var : memref<i32>) {
    omp.terminator
  }
  return
}

// -----

func @omp_sections() {
  // expected-error @below {{schedule is not a valid clause for the omp.sections operation}}
  omp.sections schedule(static, none) {
    omp.terminator
  }
  return
}

// -----

func @omp_sections() {
  // expected-error @below {{collapse is not a valid clause for the omp.sections operation}}
  omp.sections collapse(3) {
    omp.terminator
  }
  return
}

// -----

func @omp_sections() {
  // expected-error @below {{ordered is not a valid clause for the omp.sections operation}}
  omp.sections ordered(2) {
    omp.terminator
  }
  return
}

// -----

func @omp_sections() {
  // expected-error @below {{order is not a valid clause for the omp.sections operation}}
  omp.sections order(concurrent) {
    omp.terminator
  }
  return
}

// -----

func @omp_sections() {
  // expected-error @below {{failed to verify constraint: region with 1 blocks}}
  omp.sections {
    omp.section {
      omp.terminator
    }
    omp.terminator
  ^bb2:
    omp.terminator
  }
  return
}
