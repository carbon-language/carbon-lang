// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp=libiomp5 -fexceptions -fcxx-exceptions -x c++ -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp=libiomp5 -fexceptions -fcxx-exceptions -gline-tables-only -x c++ -emit-llvm %s -o - | FileCheck %s --check-prefix=TERM_DEBUG
// expected-no-diagnostics

int a;
int b;

struct St {
  St() {}
  ~St() {}
  int &get() { return a; }
};

// CHECK-LABEL: parallel_atomic_ewc
void parallel_atomic_ewc() {
#pragma omp parallel
  {
      // CHECK: invoke void @_ZN2StC1Ev(%struct.St* [[TEMP_ST_ADDR:%.+]])
      // CHECK: [[SCALAR_ADDR:%.+]] = invoke dereferenceable(4) i32* @_ZN2St3getEv(%struct.St* [[TEMP_ST_ADDR]])
      // CHECK: [[SCALAR_VAL:%.+]] = load atomic i32, i32* [[SCALAR_ADDR]] monotonic
      // CHECK: store i32 [[SCALAR_VAL]], i32* @b
      // CHECK: invoke void @_ZN2StD1Ev(%struct.St* [[TEMP_ST_ADDR]])
#pragma omp atomic read
      b = St().get();
      // CHECK: invoke void @_ZN2StC1Ev(%struct.St* [[TEMP_ST_ADDR:%.+]])
      // CHECK: [[SCALAR_ADDR:%.+]] = invoke dereferenceable(4) i32* @_ZN2St3getEv(%struct.St* [[TEMP_ST_ADDR]])
      // CHECK: [[B_VAL:%.+]] = load i32, i32* @b
      // CHECK: store atomic i32 [[B_VAL]], i32* [[SCALAR_ADDR]] monotonic
      // CHECK: invoke void @_ZN2StD1Ev(%struct.St* [[TEMP_ST_ADDR]])
#pragma omp atomic write
      St().get() = b;
      // CHECK: invoke void @_ZN2StC1Ev(%struct.St* [[TEMP_ST_ADDR:%.+]])
      // CHECK: [[SCALAR_ADDR:%.+]] = invoke dereferenceable(4) i32* @_ZN2St3getEv(%struct.St* [[TEMP_ST_ADDR]])
      // CHECK: [[B_VAL:%.+]] = load i32, i32* @b
      // CHECK: [[OLD_VAL:%.+]] = load atomic i32, i32* [[SCALAR_ADDR]] monotonic,
      // CHECK: br label %[[OMP_UPDATE:.+]]
      // CHECK: [[OMP_UPDATE]]
      // CHECK: [[OLD_PHI_VAL:%.+]] = phi i32 [ [[OLD_VAL]], %{{.+}} ], [ [[NEW_OLD_VAL:%.+]], %[[OMP_UPDATE]] ]
      // CHECK: [[NEW_VAL:%.+]] = srem i32 [[OLD_PHI_VAL]], [[B_VAL]]
      // CHECK: [[RES:%.+]] = cmpxchg i32* [[SCALAR_ADDR]], i32 [[OLD_PHI_VAL]], i32 [[NEW_VAL]] monotonic monotonic
      // CHECK: [[NEW_OLD_VAL]] = extractvalue { i32, i1 } [[RES]], 0
      // CHECK: [[COND:%.+]] = extractvalue { i32, i1 } [[RES]], 1
      // CHECK: br i1 [[COND]], label %[[OMP_DONE:.+]], label %[[OMP_UPDATE]]
      // CHECK: [[OMP_DONE]]
      // CHECK: invoke void @_ZN2StD1Ev(%struct.St* [[TEMP_ST_ADDR]])
#pragma omp atomic
      St().get() %= b;
    }
}

int &foo() { return a; }

// TERM_DEBUG-LABEL: parallel_atomic
void parallel_atomic() {
#pragma omp parallel
  {
#pragma omp atomic read
    // TERM_DEBUG-NOT: __kmpc_global_thread_num
    // TERM_DEBUG:     invoke {{.*}}foo{{.*}}()
    // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
    // TERM_DEBUG:     load atomic i32, i32* @{{.+}} monotonic, {{.*}}!dbg [[READ_LOC:![0-9]+]]
    foo() = a;
#pragma omp atomic write
    // TERM_DEBUG-NOT: __kmpc_global_thread_num
    // TERM_DEBUG:     invoke {{.*}}foo{{.*}}()
    // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
    // TERM_DEBUG-NOT: __kmpc_global_thread_num
    // TERM_DEBUG:     store atomic i32 {{%.+}}, i32* @{{.+}} monotonic, {{.*}}!dbg [[WRITE_LOC:![0-9]+]]
    a = foo();
#pragma omp atomic update
    // TERM_DEBUG-NOT: __kmpc_global_thread_num
    // TERM_DEBUG:     invoke {{.*}}foo{{.*}}()
    // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
    // TERM_DEBUG-NOT: __kmpc_global_thread_num
    // TERM_DEBUG:     atomicrmw add i32* @{{.+}}, i32 %{{.+}} monotonic, {{.*}}!dbg [[UPDATE_LOC:![0-9]+]]
    a += foo();
  }
  // TERM_DEBUG:     [[TERM_LPAD]]
  // TERM_DEBUG:     call void @__clang_call_terminate
  // TERM_DEBUG:     unreachable
}
// TERM_DEBUG-DAG: [[READ_LOC]] = !MDLocation(line: [[@LINE-25]],
// TERM_DEBUG-DAG: [[WRITE_LOC]] = !MDLocation(line: [[@LINE-20]],
// TERM_DEBUG-DAG: [[UPDATE_LOC]] = !MDLocation(line: [[@LINE-14]],
