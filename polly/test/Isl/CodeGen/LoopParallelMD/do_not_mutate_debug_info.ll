; This test checks that we do not accidently mutate the debug info when
; inserting loop parallel metadata.
; RUN: opt %loadPolly < %s  -S -polly -polly-codegen-isl -polly-ast-detect-parallel | FileCheck %s
; CHECK-NOT: !7 = metadata !{metadata !7}
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global i32* null, align 8

; Function Attrs: nounwind uwtable
define void @foo() {
entry:
  tail call void @llvm.dbg.value(metadata !18, i64 0, metadata !9, metadata !19), !dbg !20
  %0 = load i32** @A, align 8, !dbg !21, !tbaa !23
  br label %for.body, !dbg !27

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32* %0, i64 %indvars.iv, !dbg !21
  %1 = load i32* %arrayidx, align 4, !dbg !21, !tbaa !30
  %add = add nsw i32 %1, 1, !dbg !21
  store i32 %add, i32* %arrayidx, align 4, !dbg !21, !tbaa !30
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !27
  %exitcond = icmp eq i64 %indvars.iv, 1, !dbg !27
  br i1 %exitcond, label %for.end, label %for.body, !dbg !27

for.end:                                          ; preds = %for.body
  ret void, !dbg !32
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16}
!llvm.ident = !{!17}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.6.0 \001\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !12, metadata !2} ; [ DW_TAG_compile_unit ] [/local/mnt/workspace/build/tip-Release/t2.c] [DW_LANG_C99]
!1 = metadata !{metadata !"t2.c", metadata !"/local/mnt/workspace/build/tip-Release"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00foo\00foo\00\003\000\001\000\000\000\001\003", metadata !1, metadata !5, metadata !6, null, void ()* @foo, null, null, metadata !8} ; [ DW_TAG_subprogram ] [line 3] [def] [foo]
!5 = metadata !{metadata !"0x29", metadata !1}    ; [ DW_TAG_file_type ] [/local/mnt/workspace/build/tip-Release/t2.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{metadata !9}
!9 = metadata !{metadata !"0x100\00i\004\000", metadata !10, metadata !5, metadata !11} ; [ DW_TAG_auto_variable ] [i] [line 4]
!10 = metadata !{metadata !"0xb\004\003\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ] [/local/mnt/workspace/build/tip-Release/t2.c]
!11 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{metadata !13}
!13 = metadata !{metadata !"0x34\00A\00A\00\002\000\001", null, metadata !5, metadata !14, i32** @A, null} ; [ DW_TAG_variable ] [A] [line 2] [def]
!14 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!15 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!16 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!17 = metadata !{metadata !"clang version 3.6.0 "}
!18 = metadata !{i32 0}
!19 = metadata !{metadata !"0x102"}               ; [ DW_TAG_expression ]
!20 = metadata !{i32 4, i32 12, metadata !10, null}
!21 = metadata !{i32 5, i32 5, metadata !22, null}
!22 = metadata !{metadata !"0xb\004\003\001", metadata !1, metadata !10} ; [ DW_TAG_lexical_block ] [/local/mnt/workspace/build/tip-Release/t2.c]
!23 = metadata !{metadata !24, metadata !24, i64 0}
!24 = metadata !{metadata !"any pointer", metadata !25, i64 0}
!25 = metadata !{metadata !"omnipotent char", metadata !26, i64 0}
!26 = metadata !{metadata !"Simple C/C++ TBAA"}
!27 = metadata !{i32 4, i32 3, metadata !28, null}
!28 = metadata !{metadata !"0xb\002", metadata !1, metadata !29} ; [ DW_TAG_lexical_block ] [/local/mnt/workspace/build/tip-Release/t2.c]
!29 = metadata !{metadata !"0xb\001", metadata !1, metadata !22} ; [ DW_TAG_lexical_block ] [/local/mnt/workspace/build/tip-Release/t2.c]
!30 = metadata !{metadata !31, metadata !31, i64 0}
!31 = metadata !{metadata !"int", metadata !25, i64 0}
!32 = metadata !{i32 6, i32 1, metadata !4, null}
