; RUN: llc < %s -filetype=obj | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; from (at -Os):
; void foo() {
;   float a = 3.14;
;   *(int *)&a = 0;
; }
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; Function Attrs: nounwind optsize readnone uwtable
define void @foo() #0 {
entry:
  tail call void @llvm.dbg.declare(metadata float* undef, metadata !13, metadata !19), !dbg !20
  tail call void @llvm.dbg.value(metadata i32 1078523331, i64 0, metadata !13, metadata !19), !dbg !20
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !13, metadata !19), !dbg !20
; CHECK:  DW_AT_const_value [DW_FORM_sdata]    (0)
; CHECK-NEXT: DW_AT_name {{.*}}"a"
  ret void, !dbg !21
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind optsize readnone uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16, !17}
!llvm.ident = !{!18}

!0 = !{!"0x11\0012\00clang version 3.7.0 (trunk 227686)\001\00\000\00\001", !1, !2, !3, !6, !2, !2} ; [ DW_TAG_compile_unit ] [foo.c] [DW_LANG_C99]
!1 = !{!"foo.c", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0xf\00\000\0064\0064\000\000", null, null, !5} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!6 = !{!7}
!7 = !{!"0x2e\00foo\00foo\00\001\000\001\000\000\000\001\001", !8, !9, !10, null, void ()* @foo, null, null, !12} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!8 = !{!"foo.c", !""}
!9 = !{!"0x29", !8}                               ; [ DW_TAG_file_type ]
!10 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !11, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = !{null}
!12 = !{!13}
!13 = !{!"0x100\00a\002\000", !7, !9, !14}        ; [ DW_TAG_auto_variable ] [a] [line 2]
!14 = !{!"0x24\00float\000\0032\0032\000\000\004", null, null} ; [ DW_TAG_base_type ] [float] [line 0, size 32, align 32, offset 0, enc DW_ATE_float]
!15 = !{i32 2, !"Dwarf Version", i32 2}
!16 = !{i32 2, !"Debug Info Version", i32 2}
!17 = !{i32 1, !"PIC Level", i32 2}
!18 = !{!"clang version 3.7.0 (trunk 227686)"}
!19 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!20 = !MDLocation(line: 2, column: 9, scope: !7)
!21 = !MDLocation(line: 4, column: 1, scope: !7)
