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

!0 = !{!"0x11\004\00clang version 3.4 (trunk 187958) (llvm/trunk 187964)\000\00\000\00\000", !1, !2, !2, !2, !3, !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/debug-info-template-recursive.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"debug-info-template-recursive.cpp", !"/usr/local/google/home/echristo/tmp"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x34\00filters\00filters\00\0010\000\001", null, !5, !6, %class.bar* @filters, null} ; [ DW_TAG_variable ] [filters] [line 10] [def]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/debug-info-template-recursive.cpp]
!6 = !{!"0x2\00bar\009\008\008\000\000\000", !1, null, null, !7, null, null, null} ; [ DW_TAG_class_type ] [bar] [line 9, size 8, align 8, offset 0] [def] [from ]
!7 = !{!8, !31}
!8 = !{!"0x1c\00\000\000\000\000\000", null, !6, !9} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from foo<void>]
!9 = !{!"0x2\00foo<void>\005\008\008\000\000\000", !1, null, null, !10, null, !29, null} ; [ DW_TAG_class_type ] [foo<void>] [line 5, size 8, align 8, offset 0] [def] [from ]
!10 = !{!11, !19, !25}
!11 = !{!"0x1c\00\000\000\000\000\000", null, !9, !12} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from base]
!12 = !{!"0x2\00base\003\008\008\000\000\000", !1, null, null, !13, null, null, null} ; [ DW_TAG_class_type ] [base] [line 3, size 8, align 8, offset 0] [def] [from ]
!13 = !{!14}
!14 = !{!"0x2e\00base\00base\00\003\000\000\000\006\00320\000\003", !1, !12, !15, null, null, null, i32 0, !18} ; [ DW_TAG_subprogram ] [line 3] [base]
!15 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = !{null, !17}
!17 = !{!"0xf\00\000\0064\0064\000\001088", i32 0, null, !12} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from base]
!18 = !{i32 786468}
!19 = !{!"0x2e\00operator=\00operator=\00_ZN3fooIvEaSES0_\006\000\000\000\006\00257\000\006", !1, !9, !20, null, null, null, i32 0, !24} ; [ DW_TAG_subprogram ] [line 6] [private] [operator=]
!20 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !21, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = !{null, !22, !23}
!22 = !{!"0xf\00\000\0064\0064\000\001088", i32 0, null, !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from foo<void>]
!23 = !{!"0x26\00\000\000\000\000\000", null, null, !9} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from foo<void>]
!24 = !{i32 786468}
!25 = !{!"0x2e\00foo\00foo\00\005\000\000\000\006\00320\000\005", !1, !9, !26, null, null, null, i32 0, !28} ; [ DW_TAG_subprogram ] [line 5] [foo]
!26 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !27, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!27 = !{null, !22}
!28 = !{i32 786468}
!29 = !{!30}
!30 = !{!"0x2f\00T\000\000", null, null, null} ; [ DW_TAG_template_type_parameter ]
!31 = !{!"0x2e\00bar\00bar\00\009\000\000\000\006\00320\000\009", !1, !6, !32, null, null, null, i32 0, !35} ; [ DW_TAG_subprogram ] [line 9] [bar]
!32 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !33, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!33 = !{null, !34}
!34 = !{!"0xf\00\000\0064\0064\000\001088", i32 0, null, !6} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from bar]
!35 = !{i32 786468}
!36 = !{i32 2, !"Dwarf Version", i32 3}
!37 = !{i32 1, !"Debug Info Version", i32 2}
