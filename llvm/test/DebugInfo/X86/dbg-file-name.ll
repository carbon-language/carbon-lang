; RUN: llc -enable-dwarf-directory -mtriple x86_64-apple-darwin10.0.0  < %s | FileCheck %s

; Radar 8884898
; CHECK: file	1 "simple.c"

declare i32 @printf(i8*, ...) nounwind

define i32 @main() nounwind {
  ret i32 0
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12}

!1 = metadata !{metadata !"0x29", metadata !10} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\001\00LLVM build 00\001\00\000\00\000", metadata !10, metadata !11, metadata !11, metadata !9, null, null} ; [ DW_TAG_compile_unit ]
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !10, metadata !1} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x2e\00main\00main\00main\009\000\001\000\006\00256\000\000", metadata !10, metadata !1, metadata !7, null, i32 ()* @main, null, null, null} ; [ DW_TAG_subprogram ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !10, metadata !1, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !5}
!9 = metadata !{metadata !6}
!10 = metadata !{metadata !"simple.c", metadata !"/Users/manav/one/two"}
!11 = metadata !{i32 0}
!12 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
