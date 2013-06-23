; REQUIRES: object-emission

; RUN: llc -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; IR generated from the following code compiled with clang -g:
; enum e { I, J = 0xffffffffU, K = 0xf000000000000000ULL } x;

; These values were previously being truncated to -1 and 0 respectively.

; CHECK: debug_info contents
; CHECK: DW_TAG_enumerator
; CHECK: DW_TAG_enumerator
; CHECK-NEXT: DW_AT_name{{.*}} = "J"
; CHECK-NEXT: DW_AT_const_value [DW_FORM_sdata]     (4294967295)
; CHECK: DW_TAG_enumerator
; CHECK-NEXT: DW_AT_name{{.*}} = "K"
; CHECK-NEXT: DW_AT_const_value [DW_FORM_sdata]     (-1152921504606846976)

@x = common global i64 0, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !8, metadata !8, metadata !9, metadata !8, metadata !""} ; [ DW_TAG_compile_unit ] [/tmp/enum.c] [DW_LANG_C99]
!1 = metadata !{metadata !"enum.c", metadata !"/tmp"}
!2 = metadata !{metadata !3}
!3 = metadata !{i32 786436, metadata !1, null, metadata !"e", i32 1, i64 64, i64 64, i32 0, i32 0, null, metadata !4, i32 0, i32 0} ; [ DW_TAG_enumeration_type ] [e] [line 1, size 64, align 64, offset 0] [def] [from ]
!4 = metadata !{metadata !5, metadata !6, metadata !7}
!5 = metadata !{i32 786472, metadata !"I", i64 0} ; [ DW_TAG_enumerator ] [I :: 0]
!6 = metadata !{i32 786472, metadata !"J", i64 4294967295} ; [ DW_TAG_enumerator ] [J :: 4294967295]
!7 = metadata !{i32 786472, metadata !"K", i64 -1152921504606846976} ; [ DW_TAG_enumerator ] [K :: 17293822569102704640]
!8 = metadata !{i32 0}
!9 = metadata !{metadata !10}
!10 = metadata !{i32 786484, i32 0, null, metadata !"x", metadata !"x", metadata !"", metadata !11, i32 1, metadata !3, i32 0, i32 1, i64* @x, null} ; [ DW_TAG_variable ] [x] [line 1] [def]
!11 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/tmp/enum.c]
!12 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
