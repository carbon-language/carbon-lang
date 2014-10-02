; RUN: opt < %s -loop-vectorize -S -pass-remarks-missed='loop-vectorize' -pass-remarks-analysis='loop-vectorize' 2>&1 | FileCheck %s

; C/C++ code for tests
; void test(int *A, int Length) {
; #pragma clang loop vectorize(enable) interleave(enable)
;   for (int i = 0; i < Length; i++) {
;     A[i] = i;
;     if (A[i] > Length)
;       break;
;   }
; }

; void test_disabled(int *A, int Length) {
; #pragma clang loop vectorize(disable) interleave(disable)
;   for (int i = 0; i < Length; i++)
;     A[i] = i;
; }

; void test_array_bounds(int *A, int *B, int Length) {
; #pragma clang loop vectorize(enable)
;   for (int i = 0; i < Length; i++)
;     A[i] = A[B[i]];
; }

; File, line, and column should match those specified in the metadata
; CHECK: remark: source.cpp:4:5: loop not vectorized: could not determine number of loop iterations
; CHECK: remark: source.cpp:4:5: loop not vectorized: use -Rpass-analysis=loop-vectorize for more info
; CHECK: remark: source.cpp:13:5: loop not vectorized: vector width and interleave count are explicitly set to 1
; CHECK: remark: source.cpp:19:5: loop not vectorized: cannot identify array bounds
; CHECK: remark: source.cpp:19:5: loop not vectorized: use -Rpass-analysis=loop-vectorize for more info
; CHECK: warning: source.cpp:19:5: loop not vectorized: failed explicitly specified loop vectorization

; CHECK: _Z4testPii
; CHECK-NOT: x i32>
; CHECK: ret

; CHECK: _Z13test_disabledPii
; CHECK-NOT: x i32>
; CHECK: ret

; CHECK: _Z17test_array_boundsPiS_i
; CHECK-NOT: x i32>
; CHECK: ret

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind optsize ssp uwtable
define void @_Z4testPii(i32* nocapture %A, i32 %Length) #0 {
entry:
  %cmp10 = icmp sgt i32 %Length, 0, !dbg !12
  br i1 %cmp10, label %for.body, label %for.end, !dbg !12, !llvm.loop !14

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32* %A, i64 %indvars.iv, !dbg !16
  %0 = trunc i64 %indvars.iv to i32, !dbg !16
  store i32 %0, i32* %arrayidx, align 4, !dbg !16, !tbaa !18
  %cmp3 = icmp sle i32 %0, %Length, !dbg !22
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !12
  %1 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %1, %Length, !dbg !12
  %or.cond = and i1 %cmp3, %cmp, !dbg !22
  br i1 %or.cond, label %for.body, label %for.end, !dbg !22

for.end:                                          ; preds = %for.body, %entry
  ret void, !dbg !24
}

; Function Attrs: nounwind optsize ssp uwtable
define void @_Z13test_disabledPii(i32* nocapture %A, i32 %Length) #0 {
entry:
  %cmp4 = icmp sgt i32 %Length, 0, !dbg !25
  br i1 %cmp4, label %for.body, label %for.end, !dbg !25, !llvm.loop !27

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32* %A, i64 %indvars.iv, !dbg !30
  %0 = trunc i64 %indvars.iv to i32, !dbg !30
  store i32 %0, i32* %arrayidx, align 4, !dbg !30, !tbaa !18
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !25
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !25
  %exitcond = icmp eq i32 %lftr.wideiv, %Length, !dbg !25
  br i1 %exitcond, label %for.end, label %for.body, !dbg !25, !llvm.loop !27

for.end:                                          ; preds = %for.body, %entry
  ret void, !dbg !31
}

; Function Attrs: nounwind optsize ssp uwtable
define void @_Z17test_array_boundsPiS_i(i32* nocapture %A, i32* nocapture readonly %B, i32 %Length) #0 {
entry:
  %cmp9 = icmp sgt i32 %Length, 0, !dbg !32
  br i1 %cmp9, label %for.body.preheader, label %for.end, !dbg !32, !llvm.loop !34

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !35

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32* %B, i64 %indvars.iv, !dbg !35
  %0 = load i32* %arrayidx, align 4, !dbg !35, !tbaa !18
  %idxprom1 = sext i32 %0 to i64, !dbg !35
  %arrayidx2 = getelementptr inbounds i32* %A, i64 %idxprom1, !dbg !35
  %1 = load i32* %arrayidx2, align 4, !dbg !35, !tbaa !18
  %arrayidx4 = getelementptr inbounds i32* %A, i64 %indvars.iv, !dbg !35
  store i32 %1, i32* %arrayidx4, align 4, !dbg !35, !tbaa !18
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !32
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !32
  %exitcond = icmp eq i32 %lftr.wideiv, %Length, !dbg !32
  br i1 %exitcond, label %for.end.loopexit, label %for.body, !dbg !32, !llvm.loop !34

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void, !dbg !36
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = metadata !{metadata !"0x11\004\00clang version 3.5.0\001\00\006\00\002", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [./source.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"source.cpp", metadata !"."}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !7, metadata !8}
!4 = metadata !{metadata !"0x2e\00test\00test\00\001\000\001\000\006\00256\001\001", metadata !1, metadata !5, metadata !6, null, void (i32*, i32)* @_Z4testPii, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [test]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [./source.cpp]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !"0x2e\00test_disabled\00test_disabled\00\0010\000\001\000\006\00256\001\0010", metadata !1, metadata !5, metadata !6, null, void (i32*, i32)* @_Z13test_disabledPii, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 10] [def] [test_disabled]
!8 = metadata !{metadata !"0x2e\00test_array_bounds\00test_array_bounds\00\0016\000\001\000\006\00256\001\0016", metadata !1, metadata !5, metadata !6, null, void (i32*, i32*, i32)* @_Z17test_array_boundsPiS_i, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 16] [def] [test_array_bounds]
!9 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!10 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!11 = metadata !{metadata !"clang version 3.5.0"}
!12 = metadata !{i32 3, i32 8, metadata !13, null}
!13 = metadata !{metadata !"0xb\003\003\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ]
!14 = metadata !{metadata !14, metadata !15, metadata !15}
!15 = metadata !{metadata !"llvm.loop.vectorize.enable", i1 true}
!16 = metadata !{i32 4, i32 5, metadata !17, null}
!17 = metadata !{metadata !"0xb\003\0036\000", metadata !1, metadata !13} ; [ DW_TAG_lexical_block ]
!18 = metadata !{metadata !19, metadata !19, i64 0}
!19 = metadata !{metadata !"int", metadata !20, i64 0}
!20 = metadata !{metadata !"omnipotent char", metadata !21, i64 0}
!21 = metadata !{metadata !"Simple C/C++ TBAA"}
!22 = metadata !{i32 5, i32 9, metadata !23, null}
!23 = metadata !{metadata !"0xb\005\009\000", metadata !1, metadata !17} ; [ DW_TAG_lexical_block ]
!24 = metadata !{i32 8, i32 1, metadata !4, null}
!25 = metadata !{i32 12, i32 8, metadata !26, null}
!26 = metadata !{metadata !"0xb\0012\003\000", metadata !1, metadata !7} ; [ DW_TAG_lexical_block ]
!27 = metadata !{metadata !27, metadata !28, metadata !29}
!28 = metadata !{metadata !"llvm.loop.interleave.count", i32 1}
!29 = metadata !{metadata !"llvm.loop.vectorize.width", i32 1}
!30 = metadata !{i32 13, i32 5, metadata !26, null}
!31 = metadata !{i32 14, i32 1, metadata !7, null}
!32 = metadata !{i32 18, i32 8, metadata !33, null}
!33 = metadata !{metadata !"0xb\0018\003\000", metadata !1, metadata !8} ; [ DW_TAG_lexical_block ]
!34 = metadata !{metadata !34, metadata !15}
!35 = metadata !{i32 19, i32 5, metadata !33, null}
!36 = metadata !{i32 20, i32 1, metadata !8, null}
