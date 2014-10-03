; RUN: opt %loadPolly -polly-codegen-isl -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @foo(float* %A, i64 %N) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  tail call void @llvm.dbg.value(metadata !{float* %A}, i64 0, metadata !14, metadata !{metadata !"0x102"}), !dbg !15
  tail call void @llvm.dbg.value(metadata !{i64 %N}, i64 0, metadata !16, metadata !{metadata !"0x102"}), !dbg !15
  tail call void @llvm.dbg.value(metadata !17, i64 0, metadata !18, metadata !{metadata !"0x102"}), !dbg !20
  %cmp1 = icmp sgt i64 %N, 0, !dbg !20
  br i1 %cmp1, label %for.body.lr.ph, label %for.end, !dbg !20

for.body.lr.ph:                                   ; preds = %entry.split
  br label %for.body, !dbg !20

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %0 = phi i64 [ 0, %for.body.lr.ph ], [ %1, %for.body ], !dbg !21
  %arrayidx = getelementptr float* %A, i64 %0, !dbg !21
  %conv = sitofp i64 %0 to float, !dbg !21
  store float %conv, float* %arrayidx, align 4, !dbg !21
  %1 = add nsw i64 %0, 1, !dbg !20
  tail call void @llvm.dbg.value(metadata !{i64 %1}, i64 0, metadata !18, metadata !{metadata !"0x102"}), !dbg !20
  %exitcond = icmp ne i64 %1, %N, !dbg !20
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge, !dbg !20

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end, !dbg !20

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  ret void, !dbg !22
}

; CHECK: polly.split_new_and_old:

; CHECK: tail call void @llvm.dbg.value
; CHECK: tail call void @llvm.dbg.value
; CHECK: tail call void @llvm.dbg.value
; CHECK: tail call void @llvm.dbg.value
; CHECK-NOT: tail call void @llvm.dbg.value

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.5 \000\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/home/grosser/Projects/polly/git/tools/polly/loop.c] [DW_LANG_C99]
!1 = metadata !{metadata !"loop.c", metadata !"/home/grosser/Projects/polly/git/tools/polly"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00foo\00foo\00\001\000\001\000\006\00256\000\001", metadata !1, metadata !5, metadata !6, null, void (float*, i64)* @foo, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/home/grosser/Projects/polly/git/tools/polly/loop.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8, metadata !10}
!8 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from float]
!9 = metadata !{metadata !"0x24\00float\000\0032\0032\000\000\004", null, null} ; [ DW_TAG_base_type ] [float] [line 0, size 32, align 32, offset 0, enc DW_ATE_float]
!10 = metadata !{metadata !"0x24\00long int\000\0064\0064\000\000\005", null, null} ; [ DW_TAG_base_type ] [long int] [line 0, size 64, align 64, offset 0, enc DW_ATE_signed]
!11 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!12 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!13 = metadata !{metadata !"clang version 3.5 "}
!14 = metadata !{metadata !"0x101\00A\0016777217\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [A] [line 1]
!15 = metadata !{i32 1, i32 0, metadata !4, null}
!16 = metadata !{metadata !"0x101\00N\0033554433\000", metadata !4, metadata !5, metadata !10} ; [ DW_TAG_arg_variable ] [N] [line 1]
!17 = metadata !{i64 0}
!18 = metadata !{metadata !"0x100\00i\002\000", metadata !19, metadata !5, metadata !10} ; [ DW_TAG_auto_variable ] [i] [line 2]
!19 = metadata !{metadata !"0xb\002\000\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/loop.c]
!20 = metadata !{i32 2, i32 0, metadata !19, null}
!21 = metadata !{i32 3, i32 0, metadata !19, null}
!22 = metadata !{i32 4, i32 0, metadata !4, null}
