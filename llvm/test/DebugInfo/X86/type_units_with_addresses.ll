; REQUIRES: object-emission

; RUN: llc -split-dwarf=Enable -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump - | FileCheck %s

; RUN: llc -split-dwarf=Disable -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump - | FileCheck --check-prefix=SINGLE %s

; Test case built from:
;int i;
;
;template <int *I>
;struct S1 {};
;
;S1<&i> s1;
;
;template <int *I>
;struct S2_1 {};
;
;struct S2 {
;  S2_1<&i> s2_1;
;};
;
;S2 s2;
;
;template <int *I>
;struct S3_1 {};
;
;struct S3_2 {};
;
;struct S3 {
;  S3_1<&i> s3_1;
;  S3_2 s3_2;
;};
;
;S3 s3;
;
;struct S4_1 {};
;
;template <int *T>
;struct S4_2 {};
;
;struct S4 {
;  S4_1 s4_1;
;  S4_2<&::i> s4_2;
;};
;
;S4 s4;


; CHECK: .debug_info.dwo contents:

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"S1<&i>"

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"S2"
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"S2_1<&i>"

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"S3"
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"S3_1<&i>"
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_declaration
; CHECK-NEXT: DW_AT_signature

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"S4"
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_declaration
; CHECK-NEXT: DW_AT_signature
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"S4_2<&i>"

; SINGLE: .debug_info contents:

; SINGLE: DW_TAG_structure_type
; SINGLE-NEXT: DW_AT_declaration
; SINGLE-NEXT: DW_AT_signature

; SINGLE: DW_TAG_structure_type
; SINGLE-NEXT: DW_AT_declaration
; SINGLE-NEXT: DW_AT_signature

; SINGLE: DW_TAG_structure_type
; SINGLE-NEXT: DW_AT_declaration
; SINGLE-NEXT: DW_AT_signature

; SINGLE: DW_TAG_structure_type
; SINGLE-NEXT: DW_AT_declaration
; SINGLE-NEXT: DW_AT_signature

%struct.S1 = type { i8 }
%struct.S2 = type { %struct.S2_1 }
%struct.S2_1 = type { i8 }
%struct.S3 = type { %struct.S3_1, %struct.S3_2 }
%struct.S3_1 = type { i8 }
%struct.S3_2 = type { i8 }
%struct.S4 = type { %struct.S4_1, %struct.S4_2 }
%struct.S4_1 = type { i8 }
%struct.S4_2 = type { i8 }

@i = global i32 0, align 4
@a = global %struct.S1 zeroinitializer, align 1
@s2 = global %struct.S2 zeroinitializer, align 1
@s3 = global %struct.S3 zeroinitializer, align 1
@s4 = global %struct.S4 zeroinitializer, align 1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!34, !35}
!llvm.ident = !{!36}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !2, metadata !27, metadata !2, metadata !"tu.dwo", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/tu.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"tu.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !9, metadata !12, metadata !13, metadata !17, metadata !18, metadata !19, metadata !23, metadata !24}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"S1<&i>", i32 4, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, metadata !5, metadata !"_ZTS2S1IXadL_Z1iEEE"} ; [ DW_TAG_structure_type ] [S1<&i>] [line 4, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{i32 786480, null, metadata !"I", metadata !7, i32* @i, null, i32 0, i32 0} ; [ DW_TAG_template_value_parameter ]
!7 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !8} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786451, metadata !1, null, metadata !"S2", i32 11, i64 8, i64 8, i32 0, i32 0, null, metadata !10, i32 0, null, null, metadata !"_ZTS2S2"} ; [ DW_TAG_structure_type ] [S2] [line 11, size 8, align 8, offset 0] [def] [from ]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 786445, metadata !1, metadata !"_ZTS2S2", metadata !"s2_1", i32 12, i64 8, i64 8, i64 0, i32 0, metadata !"_ZTS4S2_1IXadL_Z1iEEE"} ; [ DW_TAG_member ] [s2_1] [line 12, size 8, align 8, offset 0] [from _ZTS4S2_1IXadL_Z1iEEE]
!12 = metadata !{i32 786451, metadata !1, null, metadata !"S2_1<&i>", i32 9, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, metadata !5, metadata !"_ZTS4S2_1IXadL_Z1iEEE"} ; [ DW_TAG_structure_type ] [S2_1<&i>] [line 9, size 8, align 8, offset 0] [def] [from ]
!13 = metadata !{i32 786451, metadata !1, null, metadata !"S3", i32 22, i64 16, i64 8, i32 0, i32 0, null, metadata !14, i32 0, null, null, metadata !"_ZTS2S3"} ; [ DW_TAG_structure_type ] [S3] [line 22, size 16, align 8, offset 0] [def] [from ]
!14 = metadata !{metadata !15, metadata !16}
!15 = metadata !{i32 786445, metadata !1, metadata !"_ZTS2S3", metadata !"s3_1", i32 23, i64 8, i64 8, i64 0, i32 0, metadata !"_ZTS4S3_1IXadL_Z1iEEE"} ; [ DW_TAG_member ] [s3_1] [line 23, size 8, align 8, offset 0] [from _ZTS4S3_1IXadL_Z1iEEE]
!16 = metadata !{i32 786445, metadata !1, metadata !"_ZTS2S3", metadata !"s3_2", i32 24, i64 8, i64 8, i64 8, i32 0, metadata !"_ZTS4S3_2"} ; [ DW_TAG_member ] [s3_2] [line 24, size 8, align 8, offset 8] [from _ZTS4S3_2]
!17 = metadata !{i32 786451, metadata !1, null, metadata !"S3_1<&i>", i32 18, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, metadata !5, metadata !"_ZTS4S3_1IXadL_Z1iEEE"} ; [ DW_TAG_structure_type ] [S3_1<&i>] [line 18, size 8, align 8, offset 0] [def] [from ]
!18 = metadata !{i32 786451, metadata !1, null, metadata !"S3_2", i32 20, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, null, metadata !"_ZTS4S3_2"} ; [ DW_TAG_structure_type ] [S3_2] [line 20, size 8, align 8, offset 0] [def] [from ]
!19 = metadata !{i32 786451, metadata !1, null, metadata !"S4", i32 34, i64 16, i64 8, i32 0, i32 0, null, metadata !20, i32 0, null, null, metadata !"_ZTS2S4"} ; [ DW_TAG_structure_type ] [S4] [line 34, size 16, align 8, offset 0] [def] [from ]
!20 = metadata !{metadata !21, metadata !22}
!21 = metadata !{i32 786445, metadata !1, metadata !"_ZTS2S4", metadata !"s4_1", i32 35, i64 8, i64 8, i64 0, i32 0, metadata !"_ZTS4S4_1"} ; [ DW_TAG_member ] [s4_1] [line 35, size 8, align 8, offset 0] [from _ZTS4S4_1]
!22 = metadata !{i32 786445, metadata !1, metadata !"_ZTS2S4", metadata !"s4_2", i32 36, i64 8, i64 8, i64 8, i32 0, metadata !"_ZTS4S4_2IXadL_Z1iEEE"} ; [ DW_TAG_member ] [s4_2] [line 36, size 8, align 8, offset 8] [from _ZTS4S4_2IXadL_Z1iEEE]
!23 = metadata !{i32 786451, metadata !1, null, metadata !"S4_1", i32 29, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, null, metadata !"_ZTS4S4_1"} ; [ DW_TAG_structure_type ] [S4_1] [line 29, size 8, align 8, offset 0] [def] [from ]
!24 = metadata !{i32 786451, metadata !1, null, metadata !"S4_2<&i>", i32 32, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, metadata !25, metadata !"_ZTS4S4_2IXadL_Z1iEEE"} ; [ DW_TAG_structure_type ] [S4_2<&i>] [line 32, size 8, align 8, offset 0] [def] [from ]
!25 = metadata !{metadata !26}
!26 = metadata !{i32 786480, null, metadata !"T", metadata !7, i32* @i, null, i32 0, i32 0} ; [ DW_TAG_template_value_parameter ]
!27 = metadata !{metadata !28, metadata !30, metadata !31, metadata !32, metadata !33}
!28 = metadata !{i32 786484, i32 0, null, metadata !"i", metadata !"i", metadata !"", metadata !29, i32 1, metadata !8, i32 0, i32 1, i32* @i, null} ; [ DW_TAG_variable ] [i] [line 1] [def]
!29 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/tu.cpp]
!30 = metadata !{i32 786484, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !29, i32 6, metadata !"_ZTS2S1IXadL_Z1iEEE", i32 0, i32 1, %struct.S1* @a, null} ; [ DW_TAG_variable ] [a] [line 6] [def]
!31 = metadata !{i32 786484, i32 0, null, metadata !"s2", metadata !"s2", metadata !"", metadata !29, i32 15, metadata !"_ZTS2S2", i32 0, i32 1, %struct.S2* @s2, null} ; [ DW_TAG_variable ] [s2] [line 15] [def]
!32 = metadata !{i32 786484, i32 0, null, metadata !"s3", metadata !"s3", metadata !"", metadata !29, i32 27, metadata !"_ZTS2S3", i32 0, i32 1, %struct.S3* @s3, null} ; [ DW_TAG_variable ] [s3] [line 27] [def]
!33 = metadata !{i32 786484, i32 0, null, metadata !"s4", metadata !"s4", metadata !"", metadata !29, i32 39, metadata !"_ZTS2S4", i32 0, i32 1, %struct.S4* @s4, null} ; [ DW_TAG_variable ] [s4] [line 39] [def]
!34 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!35 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!36 = metadata !{metadata !"clang version 3.5.0 "}
