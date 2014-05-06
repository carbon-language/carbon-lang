// RUN: %clang_cc1 -verify -fopenmp=libiomp5 -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -g -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix=CHECK-DEBUG %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK-DAG: %ident_t = type { i32, i32, i32, i32, i8* }
// CHECK-DAG: %struct.anon = type { i32* }
// CHECK-DAG: %struct.anon.0 = type { i8*** }
// CHECK-DAG: @.str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CHECK-DAG: @.kmpc_default_loc_2.addr = private unnamed_addr constant %ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8]* @.str, i32 0, i32 0) }
// CHECK-DEBUG-DAG: %ident_t = type { i32, i32, i32, i32, i8* }
// CHECK-DEBUG-DAG: %struct.anon = type { i32* }
// CHECK-DEBUG-DAG: %struct.anon.0 = type { i8*** }
// CHECK-DEBUG-DAG: @.str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CHECK-DEBUG-DAG: @.kmpc_default_loc_2.addr = private unnamed_addr constant %ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8]* @.str, i32 0, i32 0) }
// CHECK-DEBUG-DAG: [[LOC1:@.+]] = private unnamed_addr constant [{{.+}} x i8] c";{{.*}}parallel_codegen.cpp;main;[[@LINE+14]];9;;\00"
// CHECK-DEBUG-DAG: [[LOC2:@.+]] = private unnamed_addr constant [{{.+}} x i8] c";{{.*}}parallel_codegen.cpp;tmain;[[@LINE+7]];9;;\00"

template <class T>
void foo(T argc) {}

template <typename T>
int tmain(T argc) {
#pragma omp parallel
  foo(argc);
  return 0;
}

int main (int argc, char **argv) {
#pragma omp parallel
  foo(argc);
  return tmain(argv);
}

// CHECK-LABEL: define i32 @main(i32 %argc, i8** %argv)
// CHECK:       %agg.captured = alloca %struct.anon
// CHECK:       [[ARGC_REF:%.+]] = getelementptr inbounds %struct.anon* %agg.captured, i32 0, i32 0
// CHECK-NEXT:  store i32* %argc.addr, i32** [[ARGC_REF]]
// CHECK-NEXT:  [[BITCAST:%.+]] = bitcast %struct.anon* %agg.captured to i8*
// CHECK-NEXT:  call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...)* @__kmpc_fork_call(%ident_t* @.kmpc_default_loc_2.addr, i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, %struct.anon*)* @__captured_stmt to void (i32*, i32*, ...)*), i8* [[BITCAST]])
// CHECK-NEXT:  [[ARGV:%.+]] = load i8*** %argv.addr, align 8
// CHECK-NEXT:  [[RET:%.+]] = call i32 @_Z5tmainIPPcEiT_(i8** [[ARGV]])
// CHECK-NEXT:  ret i32 [[RET]]
// CHECK-NEXT:  }
// CHECK-DEBUG-LABEL: define i32 @main(i32 %argc, i8** %argv)
// CHECK-DEBUG-DAG:   %agg.captured = alloca %struct.anon
// CHECK-DEBUG-DAG:   %.kmpc_loc_2.addr = alloca %ident_t
// CHECK-DEBUG:       [[KMPC_LOC_VOIDPTR:%.+]] = bitcast %ident_t* %.kmpc_loc_2.addr to i8*
// CHECK-DEBUG-NEXT:  [[KMPC_DEFAULT_LOC_VOIDPTR:%.+]] = bitcast %ident_t* @.kmpc_default_loc_2.addr to i8*
// CHECK-DEBUG-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[KMPC_LOC_VOIDPTR]], i8* [[KMPC_DEFAULT_LOC_VOIDPTR]], i64 ptrtoint (%ident_t* getelementptr (%ident_t* null, i32 1) to i64), i32 8, i1 false)
// CHECK-DEBUG:       [[ARGC_REF:%.+]] = getelementptr inbounds %struct.anon* %agg.captured, i32 0, i32 0
// CHECK-DEBUG-NEXT:  store i32* %argc.addr, i32** [[ARGC_REF]]
// CHECK-DEBUG-NEXT:  [[KMPC_LOC_PSOURCE_REF:%.+]] = getelementptr inbounds %ident_t* %.kmpc_loc_2.addr, i32 0, i32 4
// CHECK-DEBUG-NEXT:  store i8* getelementptr inbounds ([{{.+}} x i8]* [[LOC1]], i32 0, i32 0), i8** [[KMPC_LOC_PSOURCE_REF]]
// CHECK-DEBUG-NEXT:  [[BITCAST:%.+]] = bitcast %struct.anon* %agg.captured to i8*
// CHECK-DEBUG-NEXT:  call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...)* @__kmpc_fork_call(%ident_t* %.kmpc_loc_2.addr, i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, %struct.anon*)* @__captured_stmt to void (i32*, i32*, ...)*), i8* [[BITCAST]])
// CHECK-DEBUG-NEXT:  [[ARGV:%.+]] = load i8*** %argv.addr, align 8
// CHECK-DEBUG-NEXT:  [[RET:%.+]] = call i32 @_Z5tmainIPPcEiT_(i8** [[ARGV]])
// CHECK-DEBUG-NEXT:  ret i32 [[RET]]
// CHECK-DEBUG-NEXT:  }

// CHECK-LABEL: define internal void @__captured_stmt(i32* %.global_tid., i32* %.bound_tid., %struct.anon* %__context)
// CHECK:       %__context.addr = alloca %struct.anon*
// CHECK:       store %struct.anon* %__context, %struct.anon** %__context.addr
// CHECK-NEXT:  [[CONTEXT_PTR:%.+]] = load %struct.anon** %__context.addr
// CHECK-NEXT:  [[ARGC_PTR_REF:%.+]] = getelementptr inbounds %struct.anon* [[CONTEXT_PTR]], i32 0, i32 0
// CHECK-NEXT:  [[ARGC_REF:%.+]] = load i32** [[ARGC_PTR_REF]]
// CHECK-NEXT:  [[ARGC:%.+]] = load i32* [[ARGC_REF]]
// CHECK-NEXT:  invoke void @_Z3fooIiEvT_(i32 [[ARGC]])
// CHECK:       ret void
// CHECK:       call void @__clang_call_terminate(i8*
// CHECK-NEXT:  unreachable
// CHECK-NEXT:  }
// CHECK-DEBUG-LABEL: define internal void @__captured_stmt(i32* %.global_tid., i32* %.bound_tid., %struct.anon* %__context)
// CHECK-DEBUG:       %__context.addr = alloca %struct.anon*
// CHECK-DEBUG:       store %struct.anon* %__context, %struct.anon** %__context.addr
// CHECK-DEBUG:       [[CONTEXT_PTR:%.+]] = load %struct.anon** %__context.addr
// CHECK-DEBUG-NEXT:  [[ARGC_PTR_REF:%.+]] = getelementptr inbounds %struct.anon* [[CONTEXT_PTR]], i32 0, i32 0
// CHECK-DEBUG-NEXT:  [[ARGC_REF:%.+]] = load i32** [[ARGC_PTR_REF]]
// CHECK-DEBUG-NEXT:  [[ARGC:%.+]] = load i32* [[ARGC_REF]]
// CHECK-DEBUG-NEXT:  invoke void @_Z3fooIiEvT_(i32 [[ARGC]])
// CHECK-DEBUG:       ret void
// CHECK-DEBUG:       call void @__clang_call_terminate(i8*
// CHECK-DEBUG-NEXT:  unreachable
// CHECK-DEBUG-NEXT:  }

// CHECK-DAG: define linkonce_odr void @_Z3fooIiEvT_(i32 %argc)
// CHECK-DAG: declare void @__kmpc_fork_call(%ident_t*, i32, void (i32*, i32*, ...)*, ...)
// CHECK-DEBUG-DAG: define linkonce_odr void @_Z3fooIiEvT_(i32 %argc)
// CHECK-DEBUG-DAG: declare void @__kmpc_fork_call(%ident_t*, i32, void (i32*, i32*, ...)*, ...)

// CHECK-LABEL: define linkonce_odr i32 @_Z5tmainIPPcEiT_(i8** %argc)
// CHECK:       %agg.captured = alloca %struct.anon.0
// CHECK:       [[ARGC_REF:%.+]] = getelementptr inbounds %struct.anon.0* %agg.captured, i32 0, i32 0
// CHECK-NEXT:  store i8*** %argc.addr, i8**** [[ARGC_REF]]
// CHECK-NEXT:  [[BITCAST:%.+]] = bitcast %struct.anon.0* %agg.captured to i8*
// CHECK-NEXT:  call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...)* @__kmpc_fork_call(%ident_t* @.kmpc_default_loc_2.addr, i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, %struct.anon.0*)* @__captured_stmt1 to void (i32*, i32*, ...)*), i8* [[BITCAST]])
// CHECK-NEXT:  ret i32 0
// CHECK-NEXT:  }
// CHECK-DEBUG-LABEL: define linkonce_odr i32 @_Z5tmainIPPcEiT_(i8** %argc)
// CHECK-DEBUG-DAG:   %agg.captured = alloca %struct.anon.0
// CHECK-DEBUG-DAG:   %.kmpc_loc_2.addr = alloca %ident_t
// CHECK-DEBUG:       [[KMPC_LOC_VOIDPTR:%.+]] = bitcast %ident_t* %.kmpc_loc_2.addr to i8*
// CHECK-DEBUG-NEXT:  [[KMPC_DEFAULT_LOC_VOIDPTR:%.+]] = bitcast %ident_t* @.kmpc_default_loc_2.addr to i8*
// CHECK-DEBUG-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[KMPC_LOC_VOIDPTR]], i8* [[KMPC_DEFAULT_LOC_VOIDPTR]], i64 ptrtoint (%ident_t* getelementptr (%ident_t* null, i32 1) to i64), i32 8, i1 false)
// CHECK-DEBUG:       [[ARGC_REF:%.+]] = getelementptr inbounds %struct.anon.0* %agg.captured, i32 0, i32 0
// CHECK-DEBUG-NEXT:  store i8*** %argc.addr, i8**** [[ARGC_REF]]
// CHECK-DEBUG-NEXT:  [[KMPC_LOC_PSOURCE_REF:%.+]] = getelementptr inbounds %ident_t* %.kmpc_loc_2.addr, i32 0, i32 4
// CHECK-DEBUG-NEXT:  store i8* getelementptr inbounds ([{{.+}} x i8]* [[LOC2]], i32 0, i32 0), i8** [[KMPC_LOC_PSOURCE_REF]]
// CHECK-DEBUG-NEXT:  [[BITCAST:%.+]] = bitcast %struct.anon.0* %agg.captured to i8*
// CHECK-DEBUG-NEXT:  call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...)* @__kmpc_fork_call(%ident_t* %.kmpc_loc_2.addr, i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, %struct.anon.0*)* @__captured_stmt1 to void (i32*, i32*, ...)*), i8* [[BITCAST]])
// CHECK-DEBUG-NEXT:  ret i32 0
// CHECK-DEBUG-NEXT:  }

// CHECK-LABEL: define internal void @__captured_stmt1(i32* %.global_tid., i32* %.bound_tid., %struct.anon.0* %__context)
// CHECK:       %__context.addr = alloca %struct.anon.0*, align 8
// CHECK:       store %struct.anon.0* %__context, %struct.anon.0** %__context.addr, align 8
// CHECK-NEXT:  [[CONTEXT_PTR:%.+]] = load %struct.anon.0** %__context.addr
// CHECK-NEXT:  [[ARGC_PTR_REF:%.+]] = getelementptr inbounds %struct.anon.0* [[CONTEXT_PTR]], i32 0, i32 0
// CHECK-NEXT:  [[ARGC_REF:%.+]] = load i8**** [[ARGC_PTR_REF]]
// CHECK-NEXT:  [[ARGC:%.+]] = load i8*** [[ARGC_REF]]
// CHECK-NEXT:  invoke void @_Z3fooIPPcEvT_(i8** [[ARGC]])
// CHECK:       ret void
// CHECK:       call void @__clang_call_terminate(i8*
// CHECK-NEXT:  unreachable
// CHECK-NEXT:  }
// CHECK-DEBUG-LABEL: define internal void @__captured_stmt1(i32* %.global_tid., i32* %.bound_tid., %struct.anon.0* %__context)
// CHECK-DEBUG:       %__context.addr = alloca %struct.anon.0*, align 8
// CHECK-DEBUG:       store %struct.anon.0* %__context, %struct.anon.0** %__context.addr, align 8
// CHECK-DEBUG:       [[CONTEXT_PTR:%.+]] = load %struct.anon.0** %__context.addr
// CHECK-DEBUG-NEXT:  [[ARGC_PTR_REF:%.+]] = getelementptr inbounds %struct.anon.0* [[CONTEXT_PTR]], i32 0, i32 0
// CHECK-DEBUG-NEXT:  [[ARGC_REF:%.+]] = load i8**** [[ARGC_PTR_REF]]
// CHECK-DEBUG-NEXT:  [[ARGC:%.+]] = load i8*** [[ARGC_REF]]
// CHECK-DEBUG-NEXT:  invoke void @_Z3fooIPPcEvT_(i8** [[ARGC]])
// CHECK-DEBUG:       ret void
// CHECK-DEBUG:       call void @__clang_call_terminate(i8*
// CHECK-DEBUG-NEXT:  unreachable
// CHECK-DEBUG-NEXT:  }

// CHECK: define linkonce_odr void @_Z3fooIPPcEvT_(i8** %argc)
// CHECK-DEBUG: define linkonce_odr void @_Z3fooIPPcEvT_(i8** %argc)

#endif
