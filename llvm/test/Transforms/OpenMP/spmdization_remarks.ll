; RUN: opt -passes=openmp-opt -pass-remarks=openmp-opt -pass-remarks-missed=openmp-opt -pass-remarks-analysis=openmp-opt -disable-output < %s 2>&1 | FileCheck %s
target triple = "nvptx64"

; CHECK: remark: llvm/test/Transforms/OpenMP/spmdization_remarks.c:13:5: Value has potential side effects preventing SPMD-mode execution. Add `__attribute__((assume("ompx_spmd_amenable")))` to the called function to override.
; CHECK: remark: llvm/test/Transforms/OpenMP/spmdization_remarks.c:15:5: Value has potential side effects preventing SPMD-mode execution. Add `__attribute__((assume("ompx_spmd_amenable")))` to the called function to override.
; CHECK: remark: llvm/test/Transforms/OpenMP/spmdization_remarks.c:11:1: Generic-mode kernel is executed with a customized state machine that requires a fallback.
; CHECK: remark: llvm/test/Transforms/OpenMP/spmdization_remarks.c:13:5: Call may contain unknown parallel regions. Use `__attribute__((assume("omp_no_parallelism")))` to override.
; CHECK: remark: llvm/test/Transforms/OpenMP/spmdization_remarks.c:15:5: Call may contain unknown parallel regions. Use `__attribute__((assume("omp_no_parallelism")))` to override.
; CHECK: remark: llvm/test/Transforms/OpenMP/spmdization_remarks.c:20:1: Transformed generic-mode kernel to SPMD-mode.


;; void unknown(void);
;; void known(void) {
;;   #pragma omp parallel
;;   {
;;     unknown();
;;   }
;; }
;; 
;; void test_fallback(void) {
;;   #pragma omp target teams
;;   {
;;     unknown();
;;     known();
;;     unknown();
;;   }
;; }
;;
;; void no_openmp(void) __attribute__((assume("omp_no_openmp")));
;; void test_no_fallback(void) {
;;   #pragma omp target teams
;;   {
;;     known();
;;     known();
;;     known();
;;     spmd_amenable();
;;   }
;; }

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@0 = private unnamed_addr constant [103 x i8] c";llvm/test/Transforms/OpenMP/spmdization_remarks.c;__omp_offloading_2a_d80d3d_test_fallback_l11;11;1;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([103 x i8], [103 x i8]* @0, i32 0, i32 0) }, align 8
@2 = private unnamed_addr constant [72 x i8] c";llvm/test/Transforms/OpenMP/spmdization_remarks.c;test_fallback;11;1;;\00", align 1
@3 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([72 x i8], [72 x i8]* @2, i32 0, i32 0) }, align 8
@4 = private unnamed_addr constant [104 x i8] c";llvm/test/Transforms/OpenMP/spmdization_remarks.c;__omp_offloading_2a_d80d3d_test_fallback_l11;11;25;;\00", align 1
@5 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([104 x i8], [104 x i8]* @4, i32 0, i32 0) }, align 8
@__omp_offloading_2a_d80d3d_test_fallback_l11_exec_mode = weak constant i8 1
@6 = private unnamed_addr constant [106 x i8] c";llvm/test/Transforms/OpenMP/spmdization_remarks.c;__omp_offloading_2a_d80d3d_test_no_fallback_l20;20;1;;\00", align 1
@7 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([106 x i8], [106 x i8]* @6, i32 0, i32 0) }, align 8
@8 = private unnamed_addr constant [75 x i8] c";llvm/test/Transforms/OpenMP/spmdization_remarks.c;test_no_fallback;20;1;;\00", align 1
@9 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([75 x i8], [75 x i8]* @8, i32 0, i32 0) }, align 8
@10 = private unnamed_addr constant [107 x i8] c";llvm/test/Transforms/OpenMP/spmdization_remarks.c;__omp_offloading_2a_d80d3d_test_no_fallback_l20;20;25;;\00", align 1
@11 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([107 x i8], [107 x i8]* @10, i32 0, i32 0) }, align 8
@__omp_offloading_2a_d80d3d_test_no_fallback_l20_exec_mode = weak constant i8 1
@12 = private unnamed_addr constant [63 x i8] c";llvm/test/Transforms/OpenMP/spmdization_remarks.c;known;4;1;;\00", align 1
@13 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 2, i32 0, i8* getelementptr inbounds ([63 x i8], [63 x i8]* @12, i32 0, i32 0) }, align 8
@G = external global i32
@llvm.compiler.used = appending global [2 x i8*] [i8* @__omp_offloading_2a_d80d3d_test_fallback_l11_exec_mode, i8* @__omp_offloading_2a_d80d3d_test_no_fallback_l20_exec_mode], section "llvm.metadata"

; Function Attrs: convergent norecurse nounwind
define weak void @__omp_offloading_2a_d80d3d_test_fallback_l11() local_unnamed_addr #0 !dbg !15 {
entry:
  %captured_vars_addrs.i.i = alloca [0 x i8*], align 8
  %0 = call i32 @__kmpc_target_init(%struct.ident_t* nonnull @1, i1 false, i1 true, i1 true) #3, !dbg !18
  %exec_user_code = icmp eq i32 %0, -1, !dbg !18
  br i1 %exec_user_code, label %user_code.entry, label %common.ret, !dbg !18

common.ret:                                       ; preds = %entry, %user_code.entry
  ret void, !dbg !19

user_code.entry:                                  ; preds = %entry
  %1 = call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @3) #3
  call void @unknown() #6, !dbg !20
  %2 = bitcast [0 x i8*]* %captured_vars_addrs.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 0, i8* nonnull %2) #3
  %3 = call i32 @__kmpc_global_thread_num(%struct.ident_t* noundef nonnull @13) #3
  %4 = getelementptr inbounds [0 x i8*], [0 x i8*]* %captured_vars_addrs.i.i, i64 0, i64 0, !dbg !23
  call void @__kmpc_parallel_51(%struct.ident_t* noundef nonnull @13, i32 %3, i32 noundef 1, i32 noundef -1, i32 noundef -1, i8* noundef bitcast (void (i32*, i32*)* @__omp_outlined__2 to i8*), i8* noundef bitcast (void (i16, i32)* @__omp_outlined__2_wrapper to i8*), i8** noundef nonnull %4, i64 noundef 0) #3, !dbg !23
  call void @llvm.lifetime.end.p0i8(i64 0, i8* nonnull %2) #3, !dbg !26
  call void @unknown() #6, !dbg !27
  call void @__kmpc_target_deinit(%struct.ident_t* nonnull @5, i1 false, i1 true) #3, !dbg !28
  br label %common.ret
}

declare i32 @__kmpc_target_init(%struct.ident_t*, i1, i1, i1) local_unnamed_addr

; Function Attrs: convergent
declare void @unknown() local_unnamed_addr #1

; Function Attrs: nounwind
define hidden void @known() local_unnamed_addr #2 !dbg !29 {
entry:
  %captured_vars_addrs = alloca [0 x i8*], align 8
  %0 = call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @13)
  %1 = getelementptr inbounds [0 x i8*], [0 x i8*]* %captured_vars_addrs, i64 0, i64 0, !dbg !30
  call void @__kmpc_parallel_51(%struct.ident_t* nonnull @13, i32 %0, i32 1, i32 -1, i32 -1, i8* bitcast (void (i32*, i32*)* @__omp_outlined__2 to i8*), i8* bitcast (void (i16, i32)* @__omp_outlined__2_wrapper to i8*), i8** nonnull %1, i64 0) #3, !dbg !30
  ret void, !dbg !31
}

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(%struct.ident_t*) local_unnamed_addr #3

declare void @__kmpc_target_deinit(%struct.ident_t*, i1, i1) local_unnamed_addr

; Function Attrs: norecurse nounwind
define weak void @__omp_offloading_2a_d80d3d_test_no_fallback_l20() local_unnamed_addr #4 !dbg !32 {
entry:
  %captured_vars_addrs.i2.i = alloca [0 x i8*], align 8
  %0 = call i32 @__kmpc_target_init(%struct.ident_t* nonnull @7, i1 false, i1 true, i1 true) #3, !dbg !33
  %exec_user_code = icmp eq i32 %0, -1, !dbg !33
  br i1 %exec_user_code, label %user_code.entry, label %common.ret, !dbg !33

common.ret:                                       ; preds = %entry, %user_code.entry
  ret void, !dbg !34

user_code.entry:                                  ; preds = %entry
  %1 = call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @9) #3
  %2 = bitcast [0 x i8*]* %captured_vars_addrs.i2.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 0, i8* nonnull %2) #3
  %3 = call i32 @__kmpc_global_thread_num(%struct.ident_t* noundef nonnull @13) #3
  %4 = getelementptr inbounds [0 x i8*], [0 x i8*]* %captured_vars_addrs.i2.i, i64 0, i64 0, !dbg !35
  call void @__kmpc_parallel_51(%struct.ident_t* noundef nonnull @13, i32 %3, i32 noundef 1, i32 noundef -1, i32 noundef -1, i8* noundef bitcast (void (i32*, i32*)* @__omp_outlined__2 to i8*), i8* noundef bitcast (void (i16, i32)* @__omp_outlined__2_wrapper to i8*), i8** noundef nonnull %4, i64 noundef 0) #3, !dbg !35
  call void @llvm.lifetime.end.p0i8(i64 0, i8* nonnull %2) #3, !dbg !39
  call void @llvm.lifetime.start.p0i8(i64 0, i8* nonnull %2) #3
  %5 = call i32 @__kmpc_global_thread_num(%struct.ident_t* noundef nonnull @13) #3
  call void @__kmpc_parallel_51(%struct.ident_t* noundef nonnull @13, i32 %5, i32 noundef 1, i32 noundef -1, i32 noundef -1, i8* noundef bitcast (void (i32*, i32*)* @__omp_outlined__2 to i8*), i8* noundef bitcast (void (i16, i32)* @__omp_outlined__2_wrapper to i8*), i8** noundef nonnull %4, i64 noundef 0) #3, !dbg !40
  call void @llvm.lifetime.end.p0i8(i64 0, i8* nonnull %2) #3, !dbg !42
  call void @llvm.lifetime.start.p0i8(i64 0, i8* nonnull %2) #3
  %6 = call i32 @__kmpc_global_thread_num(%struct.ident_t* noundef nonnull @13) #3
  call void @__kmpc_parallel_51(%struct.ident_t* noundef nonnull @13, i32 %6, i32 noundef 1, i32 noundef -1, i32 noundef -1, i8* noundef bitcast (void (i32*, i32*)* @__omp_outlined__2 to i8*), i8* noundef bitcast (void (i16, i32)* @__omp_outlined__2_wrapper to i8*), i8** noundef nonnull %4, i64 noundef 0) #3, !dbg !43
  call void @llvm.lifetime.end.p0i8(i64 0, i8* nonnull %2) #3, !dbg !45
  call void @spmd_amenable()
  call void @__kmpc_target_deinit(%struct.ident_t* nonnull @11, i1 false, i1 true) #3, !dbg !46
  br label %common.ret
}

; Function Attrs: convergent norecurse nounwind
define internal void @__omp_outlined__2(i32* noalias nocapture nofree readnone %.global_tid., i32* noalias nocapture nofree readnone %.bound_tid.) #0 !dbg !47 {
entry:
  call void @unknown() #6, !dbg !48
  ret void, !dbg !49
}

; Function Attrs: convergent norecurse nounwind
define internal void @__omp_outlined__2_wrapper(i16 zeroext %0, i32 %1) #0 !dbg !50 {
entry:
  %global_args = alloca i8**, align 8
  call void @__kmpc_get_shared_variables(i8*** nonnull %global_args) #3, !dbg !51
  call void @unknown() #6, !dbg !52
  ret void, !dbg !51
}

declare void @__kmpc_get_shared_variables(i8***) local_unnamed_addr

declare void @__kmpc_parallel_51(%struct.ident_t*, i32, i32, i32, i32, i8*, i8*, i8**, i64) local_unnamed_addr

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #5

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #5

declare void @spmd_amenable() #7

attributes #0 = { convergent norecurse nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_53" "target-features"="+ptx32,+sm_53" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_53" "target-features"="+ptx32,+sm_53" }
attributes #2 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_53" "target-features"="+ptx32,+sm_53" }
attributes #3 = { nounwind }
attributes #4 = { norecurse nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_53" "target-features"="+ptx32,+sm_53" }
attributes #5 = { argmemonly nofree nosync nounwind willreturn }
attributes #6 = { convergent nounwind }
attributes #7 = { "llvm.assume"="ompx_spmd_amenable" }

!llvm.dbg.cu = !{!0}
!omp_offload.info = !{!3, !4}
!nvvm.annotations = !{!5, !6}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 13.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: DebugDirectivesOnly, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "spmdization_remarks.c", directory: "/data/src/llvm-project")
!2 = !{}
!3 = !{i32 0, i32 42, i32 14159165, !"test_no_fallback", i32 20, i32 1}
!4 = !{i32 0, i32 42, i32 14159165, !"test_fallback", i32 11, i32 0}
!5 = !{void ()* @__omp_offloading_2a_d80d3d_test_fallback_l11, !"kernel", i32 1}
!6 = !{void ()* @__omp_offloading_2a_d80d3d_test_no_fallback_l20, !"kernel", i32 1}
!7 = !{i32 7, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 7, !"openmp", i32 50}
!11 = !{i32 7, !"openmp-device", i32 50}
!12 = !{i32 7, !"PIC Level", i32 2}
!13 = !{i32 7, !"frame-pointer", i32 2}
!14 = !{!"clang version 13.0.0"}
!15 = distinct !DISubprogram(name: "__omp_offloading_2a_d80d3d_test_fallback_l11", scope: !16, file: !16, line: 11, type: !17, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!16 = !DIFile(filename: "llvm/test/Transforms/OpenMP/spmdization_remarks.c", directory: "/data/src/llvm-project")
!17 = !DISubroutineType(types: !2)
!18 = !DILocation(line: 11, column: 1, scope: !15)
!19 = !DILocation(line: 0, scope: !15)
!20 = !DILocation(line: 13, column: 5, scope: !21, inlinedAt: !22)
!21 = distinct !DISubprogram(name: "__omp_outlined__", scope: !16, file: !16, line: 11, type: !17, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!22 = distinct !DILocation(line: 11, column: 1, scope: !15)
!23 = !DILocation(line: 4, column: 1, scope: !24, inlinedAt: !25)
!24 = distinct !DISubprogram(name: "known", scope: !16, file: !16, line: 3, type: !17, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!25 = distinct !DILocation(line: 14, column: 5, scope: !21, inlinedAt: !22)
!26 = !DILocation(line: 8, column: 1, scope: !24, inlinedAt: !25)
!27 = !DILocation(line: 15, column: 5, scope: !21, inlinedAt: !22)
!28 = !DILocation(line: 11, column: 25, scope: !15)
!29 = distinct !DISubprogram(name: "known", scope: !16, file: !16, line: 3, type: !17, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!30 = !DILocation(line: 4, column: 1, scope: !29)
!31 = !DILocation(line: 8, column: 1, scope: !29)
!32 = distinct !DISubprogram(name: "__omp_offloading_2a_d80d3d_test_no_fallback_l20", scope: !16, file: !16, line: 20, type: !17, scopeLine: 20, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!33 = !DILocation(line: 20, column: 1, scope: !32)
!34 = !DILocation(line: 0, scope: !32)
!35 = !DILocation(line: 4, column: 1, scope: !24, inlinedAt: !36)
!36 = distinct !DILocation(line: 22, column: 5, scope: !37, inlinedAt: !38)
!37 = distinct !DISubprogram(name: "__omp_outlined__1", scope: !16, file: !16, line: 20, type: !17, scopeLine: 20, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!38 = distinct !DILocation(line: 20, column: 1, scope: !32)
!39 = !DILocation(line: 8, column: 1, scope: !24, inlinedAt: !36)
!40 = !DILocation(line: 4, column: 1, scope: !24, inlinedAt: !41)
!41 = distinct !DILocation(line: 23, column: 5, scope: !37, inlinedAt: !38)
!42 = !DILocation(line: 8, column: 1, scope: !24, inlinedAt: !41)
!43 = !DILocation(line: 4, column: 1, scope: !24, inlinedAt: !44)
!44 = distinct !DILocation(line: 24, column: 5, scope: !37, inlinedAt: !38)
!45 = !DILocation(line: 8, column: 1, scope: !24, inlinedAt: !44)
!46 = !DILocation(line: 20, column: 25, scope: !32)
!47 = distinct !DISubprogram(name: "__omp_outlined__2", scope: !16, file: !16, line: 4, type: !17, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!48 = !DILocation(line: 6, column: 5, scope: !47)
!49 = !DILocation(line: 7, column: 3, scope: !47)
!50 = distinct !DISubprogram(linkageName: "__omp_outlined__2_wrapper", scope: !16, file: !16, line: 4, type: !17, scopeLine: 4, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!51 = !DILocation(line: 4, column: 1, scope: !50)
!52 = !DILocation(line: 6, column: 5, scope: !47, inlinedAt: !53)
!53 = distinct !DILocation(line: 4, column: 1, scope: !50)
