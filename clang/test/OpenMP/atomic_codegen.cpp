// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp=libiomp5 -fexceptions -fcxx-exceptions -gline-tables-only -x c++ -emit-llvm %s -o - | FileCheck %s --check-prefix=TERM_DEBUG
// expected-no-diagnostics

int a;
int &foo() { return a; }

// TERM_DEBUG-LABEL: parallel_atomic
void parallel_atomic() {
#pragma omp parallel
  {
#pragma omp atomic read
    // TERM_DEBUG-NOT: __kmpc_global_thread_num
    // TERM_DEBUG:     invoke {{.*}}foo{{.*}}()
    // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
    // TERM_DEBUG:     load atomic i32,  i32* @{{.+}} monotonic, {{.*}}!dbg [[READ_LOC:![0-9]+]]
    foo() = a;
#pragma omp atomic write
    // TERM_DEBUG-NOT: __kmpc_global_thread_num
    // TERM_DEBUG:     invoke {{.*}}foo{{.*}}()
    // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
    // TERM_DEBUG-NOT: __kmpc_global_thread_num
    // TERM_DEBUG:     store atomic i32 {{%.+}}, i32* @{{.+}} monotonic, {{.*}}!dbg [[WRITE_LOC:![0-9]+]]
    // TERM_DEBUG:     [[TERM_LPAD]]
    // TERM_DEBUG:     call void @__clang_call_terminate
    // TERM_DEBUG:     unreachable
    a = foo();
  }
}
// TERM_DEBUG-DAG: [[READ_LOC]] = !MDLocation(line: 11,
// TERM_DEBUG-DAG: [[WRITE_LOC]] = !MDLocation(line: 17,
