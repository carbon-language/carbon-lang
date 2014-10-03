; RUN: opt %loadPolly -polly-use-runtime-alias-checks=false -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -analyze < %s 2>&1| FileCheck %s

;void f(int A[], int B[]) {
;  for (int i=0; i<42; i++)
;    A[i] = B[i];
;}

; CHECK: remark: ReportAlias-01.c:2:8: The following errors keep this region from being a Scop.
; CHECK: remark: ReportAlias-01.c:3:5: Accesses to the arrays "B", "A" may access the same memory.
; CHECK: remark: ReportAlias-01.c:3:5: Invalid Scop candidate ends here.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i32* %A, i32* %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  tail call void @llvm.dbg.value(metadata !{i32* %A}, i64 0, metadata !13), !dbg !14
  tail call void @llvm.dbg.value(metadata !{i32* %B}, i64 0, metadata !15), !dbg !16
  tail call void @llvm.dbg.value(metadata !17, i64 0, metadata !18), !dbg !20
  br label %for.body, !dbg !21

for.body:                                         ; preds = %entry.split, %for.body
  %indvar = phi i64 [ 0, %entry.split ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr i32* %B, i64 %indvar, !dbg !22
  %arrayidx2 = getelementptr i32* %A, i64 %indvar, !dbg !22
  %0 = load i32* %arrayidx, align 4, !dbg !22
  store i32 %0, i32* %arrayidx2, align 4, !dbg !22
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !18), !dbg !20
  %indvar.next = add i64 %indvar, 1, !dbg !21
  %exitcond = icmp ne i64 %indvar.next, 42, !dbg !21
  br i1 %exitcond, label %for.body, label %for.end, !dbg !21

for.end:                                          ; preds = %for.body
  ret void, !dbg !23
}

declare void @llvm.dbg.declare(metadata, metadata)
declare void @llvm.dbg.value(metadata, i64, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.6.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [test/ScopDetectionDiagnostic/ReportAlias-01.c] [DW_LANG_C99]
!1 = metadata !{metadata !"ReportAlias-01.c", metadata !"test/ScopDetectionDiagnostic/"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00f\00f\00\001\000\001\000\006\00256\000\001", metadata !1, metadata !5, metadata !6, null, void (i32*, i32*)* @f, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [f]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [test/ScopDetectionDiagnostic/ReportAlias-01.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8, metadata !8}
!8 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!11 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!12 = metadata !{metadata !"clang version 3.6.0 "}
!13 = metadata !{metadata !"0x101\00A\0016777217\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [A] [line 1]
!14 = metadata !{i32 1, i32 12, metadata !4, null}
!15 = metadata !{metadata !"0x101\00B\0033554433\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [B] [line 1]
!16 = metadata !{i32 1, i32 21, metadata !4, null}
!17 = metadata !{i32 0}
!18 = metadata !{metadata !"0x100\00i\002\000", metadata !19, metadata !5, metadata !9} ; [ DW_TAG_auto_variable ] [i] [line 2]
!19 = metadata !{metadata !"0xb\002\003\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ] [test/ScopDetectionDiagnostic/ReportAlias-01.c]
!20 = metadata !{i32 2, i32 12, metadata !19, null}
!21 = metadata !{i32 2, i32 8, metadata !19, null}
!22 = metadata !{i32 3, i32 5, metadata !19, null}
!23 = metadata !{i32 4, i32 1, metadata !4, null}
