; RUN: llc -mtriple=x86_64-linux < %s -filetype=obj | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; test that we add DW_AT_inline even when we only have concrete out of line
; instances.

; first check that we have a TAG_subprogram at a given offset and it has
; AT_inline.

; CHECK: DW_TAG_class_type
; CHECK:   DW_TAG_subprogram
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_MIPS_linkage_name {{.*}} "_ZN12nsAutoRefCntaSEi"

; CHECK: DW_TAG_class_type
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_MIPS_linkage_name {{.*}} "_ZN17nsAutoRefCnt7ReleaseEv"
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}} "~nsAutoRefCnt"

; CHECK: DW_TAG_subprogram
; CHECK-NEXT:     DW_AT_{{.*}}linkage_name {{.*}}D2
; CHECK-NEXT:     DW_AT_specification {{.*}} "~nsAutoRefCnt"
; CHECK-NEXT:     DW_AT_inline
; CHECK-NOT:      DW_AT
; CHECK: DW_TAG
; CHECK: DW_TAG_subprogram
; CHECK-NEXT:     DW_AT_{{.*}}linkage_name {{.*}}D1
; CHECK-NEXT:     DW_AT_specification {{.*}} "~nsAutoRefCnt"
; CHECK-NEXT:     DW_AT_inline
; CHECK-NOT:     DW_AT
; CHECK: [[D1_THIS_ABS:.*]]: DW_TAG_formal_parameter

; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_specification {{.*}} "_ZN17nsAutoRefCnt7ReleaseEv"
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: NULL
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} "_ZN12nsAutoRefCntaSEi"
; CHECK-NOT: NULL
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} "_ZN17nsAutoRefCntD1Ev"
; CHECK-NOT: NULL
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} "_ZN17nsAutoRefCntD2Ev"

; and then that a TAG_subprogram refers to it with AT_abstract_origin.

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_abstract_origin {{.*}} "_ZN17nsAutoRefCntD1Ev"
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_abstract_origin {{.*}} {[[D1_THIS_ABS]]} "this"
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} "_ZN17nsAutoRefCntD2Ev"


define i32 @_ZN17nsAutoRefCnt7ReleaseEv() {
entry:
  store i32 1, i32* null, align 4, !dbg !50
  tail call void @_Z8moz_freePv(i8* null) nounwind, !dbg !54
  ret i32 0
}

define void @_ZN17nsAutoRefCntD1Ev() {
entry:
  tail call void @_Z8moz_freePv(i8* null) nounwind, !dbg !57
  ret void
}

declare void @_Z8moz_freePv(i8*)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!60}

!0 = metadata !{metadata !"0x11\004\00clang version 3.1 ()\001\00\000\00\000", metadata !59, metadata !1, metadata !1, metadata !3, metadata !47,  metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5, metadata !23, metadata !27, metadata !31}
!5 = metadata !{metadata !"0x2e\00Release\00Release\00_ZN17nsAutoRefCnt7ReleaseEv\0014\000\001\000\006\00256\001\0014", metadata !6, null, metadata !7, null, i32 ()* @_ZN17nsAutoRefCnt7ReleaseEv , null, metadata !12, metadata !20} ; [ DW_TAG_subprogram ] [line 14] [def] [Release]
!6 = metadata !{metadata !"0x29", metadata !59} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !10}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", i32 0, null, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{metadata !"0x13\00nsAutoRefCnt\0010\000\000\000\004\000", metadata !59, null, null, null, null, null, null} ; [ DW_TAG_structure_type ] [nsAutoRefCnt] [line 10, size 0, align 0, offset 0] [decl] [from ]
!12 = metadata !{metadata !"0x2e\00Release\00Release\00_ZN17nsAutoRefCnt7ReleaseEv\0011\000\000\000\006\00256\001\0011", metadata !6, metadata !13, metadata !7, null, null, null, i32 0, metadata !18} ; [ DW_TAG_subprogram ]
!13 = metadata !{metadata !"0x2\00nsAutoRefCnt\0010\008\008\000\000\000", metadata !59, null, null, metadata !14, null, null} ; [ DW_TAG_class_type ]
!14 = metadata !{metadata !12, metadata !15}
!15 = metadata !{metadata !"0x2e\00~nsAutoRefCnt\00~nsAutoRefCnt\00\0012\000\000\000\006\00256\001\0012", metadata !6, metadata !13, metadata !16, null, null, null, i32 0, metadata !18} ; [ DW_TAG_subprogram ]
!16 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !17, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = metadata !{null, metadata !10}
!18 = metadata !{}
!20 = metadata !{metadata !22}
!22 = metadata !{metadata !"0x101\00this\0016777230\0064", metadata !5, metadata !6, metadata !10} ; [ DW_TAG_arg_variable ]
!23 = metadata !{metadata !"0x2e\00~nsAutoRefCnt\00~nsAutoRefCnt\00_ZN17nsAutoRefCntD1Ev\0018\000\001\000\006\00256\001\0018", metadata !6, null, metadata !16, null, void ()* @_ZN17nsAutoRefCntD1Ev, null, metadata !15, metadata !24} ; [ DW_TAG_subprogram ] [line 18] [def] [~nsAutoRefCnt]
!24 = metadata !{metadata !26}
!26 = metadata !{metadata !"0x101\00this\0016777234\0064", metadata !23, metadata !6, metadata !10} ; [ DW_TAG_arg_variable ]
!27 = metadata !{metadata !"0x2e\00~nsAutoRefCnt\00~nsAutoRefCnt\00_ZN17nsAutoRefCntD2Ev\0018\000\001\000\006\00256\001\0018", metadata !6, null, metadata !16, null, i32* null, null, metadata !15, metadata !28} ; [ DW_TAG_subprogram ] [line 18] [def] [~nsAutoRefCnt]
!28 = metadata !{metadata !30}
!30 = metadata !{metadata !"0x101\00this\0016777234\0064", metadata !27, metadata !6, metadata !10} ; [ DW_TAG_arg_variable ]
!31 = metadata !{metadata !"0x2e\00operator=\00operator=\00_ZN12nsAutoRefCntaSEi\004\000\001\000\006\00256\001\004", metadata !6, null, metadata !32, null, null, null, metadata !36, metadata !43} ; [ DW_TAG_subprogram ] [line 4] [def] [operator=]
!32 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !33, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!33 = metadata !{metadata !9, metadata !34, metadata !9}
!34 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", i32 0, null, metadata !35} ; [ DW_TAG_pointer_type ]
!35 = metadata !{metadata !"0x13\00nsAutoRefCnt\002\000\000\000\004\000", metadata !59, null, null, null, null, null, null} ; [ DW_TAG_structure_type ] [nsAutoRefCnt] [line 2, size 0, align 0, offset 0] [decl] [from ]
!36 = metadata !{metadata !"0x2e\00operator=\00operator=\00_ZN12nsAutoRefCntaSEi\004\000\000\000\006\00256\001\004", metadata !6, metadata !37, metadata !32, null, null, null, i32 0, metadata !18} ; [ DW_TAG_subprogram ]
!37 = metadata !{metadata !"0x2\00nsAutoRefCnt\002\0032\0032\000\000\000", metadata !59, null, null, metadata !38, null, null, null} ; [ DW_TAG_class_type ] [nsAutoRefCnt] [line 2, size 32, align 32, offset 0] [def] [from ]
!38 = metadata !{metadata !39, metadata !40, metadata !36}
!39 = metadata !{metadata !"0xd\00mValue\007\0032\0032\000\000", metadata !59, metadata !37, metadata !9} ; [ DW_TAG_member ]
!40 = metadata !{metadata !"0x2e\00nsAutoRefCnt\00nsAutoRefCnt\00\003\000\000\000\006\00256\001\003", metadata !6, metadata !37, metadata !41, null, null, null, i32 0, metadata !18} ; [ DW_TAG_subprogram ]
!41 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !42, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!42 = metadata !{null, metadata !34}
!43 = metadata !{metadata !45, metadata !46}
!45 = metadata !{metadata !"0x101\00this\0016777220\0064", metadata !31, metadata !6, metadata !34} ; [ DW_TAG_arg_variable ]
!46 = metadata !{metadata !"0x101\00aValue\0033554436\000", metadata !31, metadata !6, metadata !9} ; [ DW_TAG_arg_variable ]
!47 = metadata !{metadata !49}
!49 = metadata !{metadata !"0x34\00mRefCnt\00mRefCnt\00\009\000\001", null, metadata !6, metadata !37, i32* null, null} ; [ DW_TAG_variable ]
!50 = metadata !{i32 5, i32 5, metadata !51, metadata !52}
!51 = metadata !{metadata !"0xb\004\0029\002", metadata !6, metadata !31} ; [ DW_TAG_lexical_block ]
!52 = metadata !{i32 15, i32 0, metadata !53, null}
!53 = metadata !{metadata !"0xb\0014\0034\000", metadata !6, metadata !5} ; [ DW_TAG_lexical_block ]
!54 = metadata !{i32 19, i32 3, metadata !55, metadata !56}
!55 = metadata !{metadata !"0xb\0018\0041\001", metadata !6, metadata !27} ; [ DW_TAG_lexical_block ]
!56 = metadata !{i32 18, i32 41, metadata !23, metadata !52}
!57 = metadata !{i32 19, i32 3, metadata !55, metadata !58}
!58 = metadata !{i32 18, i32 41, metadata !23, null}
!59 = metadata !{metadata !"nsAutoRefCnt.ii", metadata !"/Users/espindola/mozilla-central/obj-x86_64-apple-darwin11.2.0/netwerk/base/src"}
!60 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
