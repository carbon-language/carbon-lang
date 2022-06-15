; RUN: opt -S -passes=openmp-opt -openmp-ir-builder-optimistic-attributes -pass-remarks=openmp-opt -openmp-print-gpu-kernels < %s | FileCheck %s
; RUN: opt -S -passes=openmp-opt -pass-remarks=openmp-opt -openmp-print-gpu-kernels < %s | FileCheck %s

; C input used for this test:

; void bar(void) {
;     #pragma omp parallel
;     { }
; }
; void foo(void) {
;   #pragma omp target teams
;   {
;     #pragma omp parallel
;     {}
;     bar();
;     unknown();
;     #pragma omp parallel
;     {}
;   }
; }

; Verify we replace the function pointer uses for the first and last outlined
; region (1 and 3) but not for the middle one (2) because it could be called from
; another kernel.

; CHECK-DAG: @__omp_outlined__1_wrapper.ID = private constant i8 undef
; CHECK-DAG: @__omp_outlined__2_wrapper.ID = private constant i8 undef

; CHECK-DAG:   icmp eq void (i16, i32)* %worker.work_fn.addr_cast, bitcast (i8* @__omp_outlined__1_wrapper.ID to void (i16, i32)*)
; CHECK-DAG:   icmp eq void (i16, i32)* %worker.work_fn.addr_cast, bitcast (i8* @__omp_outlined__2_wrapper.ID to void (i16, i32)*)


; CHECK-DAG:   call void @__kmpc_parallel_51(%struct.ident_t* @1, i32 %{{.*}}, i32 1, i32 -1, i32 -1, i8* bitcast (void (i32*, i32*)* @__omp_outlined__1 to i8*), i8* @__omp_outlined__1_wrapper.ID, i8** %{{.*}}, i64 0)
; CHECK-DAG:   call void @__kmpc_parallel_51(%struct.ident_t* @1, i32 %{{.*}}, i32 1, i32 -1, i32 -1, i8* bitcast (void (i32*, i32*)* @__omp_outlined__2 to i8*), i8* @__omp_outlined__2_wrapper.ID, i8** %{{.*}}, i64 0)
; CHECK-DAG:   call void @__kmpc_parallel_51(%struct.ident_t* @2, i32 %{{.*}}, i32 1, i32 -1, i32 -1, i8* bitcast (void (i32*, i32*)* @__omp_outlined__3 to i8*), i8* bitcast (void (i16, i32)* @__omp_outlined__3_wrapper to i8*), i8** %{{.*}}, i64 0)


%struct.ident_t = type { i32, i32, i32, i32, i8* }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8
@__omp_offloading_10301_87b2c_foo_l7_exec_mode = weak constant i8 1
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 2, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8
@llvm.compiler.used = appending global [1 x i8*] [i8* @__omp_offloading_10301_87b2c_foo_l7_exec_mode], section "llvm.metadata"

define weak void @__omp_offloading_10301_87b2c_foo_l7() {
entry:
  %.zero.addr = alloca i32, align 4
  %.threadid_temp. = alloca i32, align 4
  store i32 0, i32* %.zero.addr, align 4
  %0 = call i32 @__kmpc_target_init(%struct.ident_t* @1, i8 1, i1 true, i1 true)
  %exec_user_code = icmp eq i32 %0, -1
  br i1 %exec_user_code, label %user_code.entry, label %worker.exit

user_code.entry:                                  ; preds = %entry
  %1 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @1)
  store i32 %1, i32* %.threadid_temp., align 4
  call void @__omp_outlined__(i32* %.threadid_temp., i32* %.zero.addr)
  call void @__kmpc_target_deinit(%struct.ident_t* @1, i8 1, i1 true)
  ret void

worker.exit:                                      ; preds = %entry
  ret void
}

declare i32 @__kmpc_target_init(%struct.ident_t*, i8, i1, i1)
declare void @unknown()

define internal void @__omp_outlined__(i32* noalias %.global_tid., i32* noalias %.bound_tid.) {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %captured_vars_addrs = alloca [0 x i8*], align 8
  %captured_vars_addrs1 = alloca [0 x i8*], align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  %0 = load i32*, i32** %.global_tid..addr, align 8
  %1 = load i32, i32* %0, align 4
  %2 = bitcast [0 x i8*]* %captured_vars_addrs to i8**
  call void @__kmpc_parallel_51(%struct.ident_t* @1, i32 %1, i32 1, i32 -1, i32 -1, i8* bitcast (void (i32*, i32*)* @__omp_outlined__1 to i8*), i8* bitcast (void (i16, i32)* @__omp_outlined__1_wrapper to i8*), i8** %2, i64 0)
  call void @bar()
  call void @unknown()
  %3 = bitcast [0 x i8*]* %captured_vars_addrs1 to i8**
  call void @__kmpc_parallel_51(%struct.ident_t* @1, i32 %1, i32 1, i32 -1, i32 -1, i8* bitcast (void (i32*, i32*)* @__omp_outlined__2 to i8*), i8* bitcast (void (i16, i32)* @__omp_outlined__2_wrapper to i8*), i8** %3, i64 0)
  ret void
}

define internal void @__omp_outlined__1(i32* noalias %.global_tid., i32* noalias %.bound_tid.) {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  ret void
}

define internal void @__omp_outlined__1_wrapper(i16 zeroext %0, i32 %1) {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca i8**, align 8
  store i32 0, i32* %.zero.addr, align 4
  store i16 %0, i16* %.addr, align 2
  store i32 %1, i32* %.addr1, align 4
  call void @__kmpc_get_shared_variables(i8*** %global_args)
  call void @__omp_outlined__1(i32* %.addr1, i32* %.zero.addr)
  ret void
}

declare void @__kmpc_get_shared_variables(i8***)

declare void @__kmpc_parallel_51(%struct.ident_t*, i32, i32, i32, i32, i8*, i8*, i8**, i64)

define hidden void @bar() {
entry:
  %captured_vars_addrs = alloca [0 x i8*], align 8
  %0 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @2)
  %1 = bitcast [0 x i8*]* %captured_vars_addrs to i8**
  call void @__kmpc_parallel_51(%struct.ident_t* @2, i32 %0, i32 1, i32 -1, i32 -1, i8* bitcast (void (i32*, i32*)* @__omp_outlined__3 to i8*), i8* bitcast (void (i16, i32)* @__omp_outlined__3_wrapper to i8*), i8** %1, i64 0)
  ret void
}

define internal void @__omp_outlined__2(i32* noalias %.global_tid., i32* noalias %.bound_tid.) {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  ret void
}

define internal void @__omp_outlined__2_wrapper(i16 zeroext %0, i32 %1) {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca i8**, align 8
  store i32 0, i32* %.zero.addr, align 4
  store i16 %0, i16* %.addr, align 2
  store i32 %1, i32* %.addr1, align 4
  call void @__kmpc_get_shared_variables(i8*** %global_args)
  call void @__omp_outlined__2(i32* %.addr1, i32* %.zero.addr)
  ret void
}

declare i32 @__kmpc_global_thread_num(%struct.ident_t*)

declare void @__kmpc_target_deinit(%struct.ident_t*, i8, i1)

define internal void @__omp_outlined__3(i32* noalias %.global_tid., i32* noalias %.bound_tid.) {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  ret void
}

define internal void @__omp_outlined__3_wrapper(i16 zeroext %0, i32 %1) {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca i8**, align 8
  store i32 0, i32* %.zero.addr, align 4
  store i16 %0, i16* %.addr, align 2
  store i32 %1, i32* %.addr1, align 4
  call void @__kmpc_get_shared_variables(i8*** %global_args)
  call void @__omp_outlined__3(i32* %.addr1, i32* %.zero.addr)
  ret void
}

!omp_offload.info = !{!0}
!nvvm.annotations = !{!1}
!llvm.module.flags = !{!2, !3}

!0 = !{i32 0, i32 66305, i32 555956, !"foo", i32 7, i32 0}
!1 = !{void ()* @__omp_offloading_10301_87b2c_foo_l7, !"kernel", i32 1}
!2 = !{i32 7, !"openmp", i32 50}
!3 = !{i32 7, !"openmp-device", i32 50}
