; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; This was pulled from clang's debug-info-template-recursive.cpp test.
; class base { };

; template <class T> class foo : public base  {
;   void operator=(const foo r) { }
; };

; class bar : public foo<void> { };
; bar filters;

; CHECK: DW_TAG_template_type_parameter [{{.*}}]
; CHECK-NEXT: DW_AT_name{{.*}}"T"
; CHECK-NOT: DW_AT_type
; CHECK: NULL

%class.bar = type { i8 }

@filters = global %class.bar zeroinitializer, align 1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!36, !37}

!0 = metadata !{metadata !"0x11\004\00clang version 3.4 (trunk 187958) (llvm/trunk 187964)\000\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !2, metadata !3, metadata !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/debug-info-template-recursive.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"debug-info-template-recursive.cpp", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x34\00filters\00filters\00\0010\000\001", null, metadata !5, metadata !6, %class.bar* @filters, null} ; [ DW_TAG_variable ] [filters] [line 10] [def]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/debug-info-template-recursive.cpp]
!6 = metadata !{metadata !"0x2\00bar\009\008\008\000\000\000", metadata !1, null, null, metadata !7, null, null, null} ; [ DW_TAG_class_type ] [bar] [line 9, size 8, align 8, offset 0] [def] [from ]
!7 = metadata !{metadata !8, metadata !31}
!8 = metadata !{metadata !"0x1c\00\000\000\000\000\000", null, metadata !6, metadata !9} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from foo<void>]
!9 = metadata !{metadata !"0x2\00foo<void>\005\008\008\000\000\000", metadata !1, null, null, metadata !10, null, metadata !29, null} ; [ DW_TAG_class_type ] [foo<void>] [line 5, size 8, align 8, offset 0] [def] [from ]
!10 = metadata !{metadata !11, metadata !19, metadata !25}
!11 = metadata !{metadata !"0x1c\00\000\000\000\000\000", null, metadata !9, metadata !12} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from base]
!12 = metadata !{metadata !"0x2\00base\003\008\008\000\000\000", metadata !1, null, null, metadata !13, null, null, null} ; [ DW_TAG_class_type ] [base] [line 3, size 8, align 8, offset 0] [def] [from ]
!13 = metadata !{metadata !14}
!14 = metadata !{metadata !"0x2e\00base\00base\00\003\000\000\000\006\00320\000\003", metadata !1, metadata !12, metadata !15, null, null, null, i32 0, metadata !18} ; [ DW_TAG_subprogram ] [line 3] [base]
!15 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{null, metadata !17}
!17 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", i32 0, null, metadata !12} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from base]
!18 = metadata !{i32 786468}
!19 = metadata !{metadata !"0x2e\00operator=\00operator=\00_ZN3fooIvEaSES0_\006\000\000\000\006\00257\000\006", metadata !1, metadata !9, metadata !20, null, null, null, i32 0, metadata !24} ; [ DW_TAG_subprogram ] [line 6] [private] [operator=]
!20 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !21, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = metadata !{null, metadata !22, metadata !23}
!22 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", i32 0, null, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from foo<void>]
!23 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, null, metadata !9} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from foo<void>]
!24 = metadata !{i32 786468}
!25 = metadata !{metadata !"0x2e\00foo\00foo\00\005\000\000\000\006\00320\000\005", metadata !1, metadata !9, metadata !26, null, null, null, i32 0, metadata !28} ; [ DW_TAG_subprogram ] [line 5] [foo]
!26 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !27, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!27 = metadata !{null, metadata !22}
!28 = metadata !{i32 786468}
!29 = metadata !{metadata !30}
!30 = metadata !{metadata !"0x2f\00T\000\000", null, null, null} ; [ DW_TAG_template_type_parameter ]
!31 = metadata !{metadata !"0x2e\00bar\00bar\00\009\000\000\000\006\00320\000\009", metadata !1, metadata !6, metadata !32, null, null, null, i32 0, metadata !35} ; [ DW_TAG_subprogram ] [line 9] [bar]
!32 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !33, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!33 = metadata !{null, metadata !34}
!34 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", i32 0, null, metadata !6} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from bar]
!35 = metadata !{i32 786468}
!36 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
!37 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
