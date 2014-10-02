; For http://llvm.org/bugs/show_bug.cgi?id=12942
;   There are two CUs coming from /tmp/foo.c in this module. Make sure it doesn't
;   blow llc up and produces something reasonable.
;

; REQUIRES: object-emission

; RUN: %llc_dwarf %s -o %t -filetype=obj -O0
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; ModuleID = 'test.bc'

@str = private unnamed_addr constant [4 x i8] c"FOO\00"
@str1 = private unnamed_addr constant [6 x i8] c"Main!\00"

define void @foo() nounwind {
entry:
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([4 x i8]* @str, i32 0, i32 0)), !dbg !23
  ret void, !dbg !25
}

declare i32 @puts(i8* nocapture) nounwind

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %argc}, i64 0, metadata !21, metadata !{metadata !"0x102"}), !dbg !26
  tail call void @llvm.dbg.value(metadata !{i8** %argv}, i64 0, metadata !22, metadata !{metadata !"0x102"}), !dbg !27
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([6 x i8]* @str1, i32 0, i32 0)), !dbg !28
  tail call void @foo() nounwind, !dbg !30
  ret i32 0, !dbg !31
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0, !9}
!llvm.module.flags = !{!33}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.2 (trunk 156513)\001\00\000\00\001", metadata !32, metadata !1, metadata !1, metadata !3, metadata !1, metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00foo\00foo\00\005\000\001\000\006\00256\001\005", metadata !32, metadata !6, metadata !7, null, void ()* @foo, null, null, metadata !1} ; [ DW_TAG_subprogram ]
!6 = metadata !{metadata !"0x29", metadata !32} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null}
!9 = metadata !{metadata !"0x11\0012\00clang version 3.2 (trunk 156513)\001\00\000\00\001", metadata !32, metadata !1, metadata !1, metadata !10, metadata !1, metadata !1} ; [ DW_TAG_compile_unit ]
!10 = metadata !{metadata !12}
!12 = metadata !{metadata !"0x2e\00main\00main\00\0011\000\001\000\006\00256\001\0011", metadata !32, metadata !6, metadata !13, null, i32 (i32, i8**)* @main, null, null, metadata !19} ; [ DW_TAG_subprogram ]
!13 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = metadata !{metadata !15, metadata !15, metadata !16}
!15 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!16 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, null, metadata !17} ; [ DW_TAG_pointer_type ]
!17 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, null, metadata !18} ; [ DW_TAG_pointer_type ]
!18 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ]
!19 = metadata !{metadata !21, metadata !22}
!21 = metadata !{metadata !"0x101\00argc\0016777227\000", metadata !12, metadata !6, metadata !15} ; [ DW_TAG_arg_variable ]
!22 = metadata !{metadata !"0x101\00argv\0033554443\000", metadata !12, metadata !6, metadata !16} ; [ DW_TAG_arg_variable ]
!23 = metadata !{i32 6, i32 3, metadata !24, null}
!24 = metadata !{metadata !"0xb\005\0016\000", metadata !32, metadata !5} ; [ DW_TAG_lexical_block ]
!25 = metadata !{i32 7, i32 1, metadata !24, null}
!26 = metadata !{i32 11, i32 14, metadata !12, null}
!27 = metadata !{i32 11, i32 26, metadata !12, null}
!28 = metadata !{i32 12, i32 3, metadata !29, null}
!29 = metadata !{metadata !"0xb\0011\0034\000", metadata !32, metadata !12} ; [ DW_TAG_lexical_block ]
!30 = metadata !{i32 13, i32 3, metadata !29, null}
!31 = metadata !{i32 14, i32 3, metadata !29, null}
!32 = metadata !{metadata !"foo.c", metadata !"/tmp"}

; This test is simple to be cross platform (many targets don't yet have
; sufficiently good DWARF emission and/or dumping)
; CHECK: {{DW_TAG_compile_unit}}
; CHECK: {{foo\.c}}

!33 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
