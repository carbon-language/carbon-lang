; RUN: llc -mtriple=x86_64-apple-darwin12 -filetype=obj < %s \
; RUN:    | llvm-dwarfdump -debug-dump=info - | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.7"
; Radar 9511391

; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_const_value [DW_FORM_sdata]   (42)
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "i"

define i32 @foo() nounwind uwtable readnone optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !8, i64 0, metadata !6, metadata !{metadata !"0x102"}), !dbg !9
  ret i32 42, !dbg !10
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.0 (trunk 132191)\001\00\000\00\000", metadata !13, metadata !14, metadata !14, metadata !11, null,  null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !"0x2e\00foo\00foo\00\001\000\001\000\006\000\001\000", metadata !13, metadata !2, metadata !3, null, i32 ()* @foo, null, null, metadata !12} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 0] [foo]
!2 = metadata !{metadata !"0x29", metadata !13} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !13, metadata !2, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !0} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x100\00i\002\000", metadata !7, metadata !2, metadata !5} ; [ DW_TAG_auto_variable ]
!7 = metadata !{metadata !"0xb\001\0011\000", metadata !13, metadata !1} ; [ DW_TAG_lexical_block ]
!8 = metadata !{i32 42}
!9 = metadata !{i32 2, i32 12, metadata !7, null}
!10 = metadata !{i32 3, i32 2, metadata !7, null}
!11 = metadata !{metadata !1}
!12 = metadata !{metadata !6}
!13 = metadata !{metadata !"a.c", metadata !"/private/tmp"}
!14 = metadata !{i32 0}
!15 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
