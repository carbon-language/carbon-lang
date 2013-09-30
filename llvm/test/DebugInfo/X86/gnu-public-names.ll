; RUN: llc -mtriple=x86_64-pc-linux-gnu -generate-gnu-dwarf-pub-sections < %s | FileCheck -check-prefix=ASM %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu -generate-gnu-dwarf-pub-sections -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s
; ModuleID = 'dwarf-public-names.cpp'
;
; Generated from:
;
; struct C {
;   void member_function();
;   static int static_member_function();
;   static int static_member_variable;
; };
;
; int C::static_member_variable = 0;
;
; void C::member_function() {
;   static_member_variable = 0;
; }
;
; int C::static_member_function() {
;   return static_member_variable;
; }
;
; C global_variable;
;
; int global_function() {
;   return -1;
; }
;
; namespace ns {
;   void global_namespace_function() {
;     global_variable.member_function();
;   }
;   int global_namespace_variable = 1;
; }

; ASM: .section        .debug_gnu_pubnames
; ASM: .byte   32                      # Kind: VARIABLE, EXTERNAL
; ASM-NEXT: .asciz  "global_namespace_variable" # External Name

; ASM: .section        .debug_gnu_pubtypes
; ASM: .byte   16                      # Kind: TYPE, EXTERNAL
; ASM-NEXT: .asciz  "C"                     # External Name

; CHECK: .debug_info contents:
; CHECK: DW_AT_GNU_pubnames [DW_FORM_sec_offset]   (0x00000000)
; CHECK: DW_AT_GNU_pubtypes [DW_FORM_sec_offset]   (0x00000000)
; CHECK: 0x0000002e: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name {{.*}} "int"
; CHECK: 0x0000003a: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}} "C"
; CHECK: 0x0000004e: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "member_function"
; CHECK: 0x00000060: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "static_member_function"
; CHECK: 0x00000084: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "global_variable"
; CHECK: 0x000000a0: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "global_namespace_variable"
; CHECK: 0x000000af: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "global_namespace_function"
; CHECK: 0x000000ca: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_specification {{.*}}0x0000004e}
; CHECK: 0x000000f2: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_specification {{.*}}0x00000060}
; CHECK: 0x00000109: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "global_function"

; CHECK-LABEL: .debug_gnu_pubnames contents:
; CHECK-NEXT: Length:                175
; CHECK-NEXT: Version:               2
; CHECK-NEXT: Offset in .debug_info: 0
; CHECK-NEXT: Size:                  327
; CHECK-NEXT: Offset     Linkage  Kind     Name
; CHECK-DAG:  0x00000099 EXTERNAL TYPE     "ns"
; CHECK-DAG:  0x000000a0 EXTERNAL VARIABLE "global_namespace_variable"
; CHECK-DAG:  0x000000af EXTERNAL FUNCTION "global_namespace_function"
; CHECK-DAG:  0x000000f2 STATIC   FUNCTION "static_member_function"
; CHECK-DAG:  0x00000084 EXTERNAL VARIABLE "global_variable"
; CHECK-DAG:  0x00000109 EXTERNAL FUNCTION "global_function"
; CHECK-DAG:  0x000000ca STATIC   FUNCTION "member_function"

; CHECK-LABEL: debug_gnu_pubtypes contents:
; CHECK-NEXT: Length:
; CHECK-NEXT: Version:
; CHECK-NEXT: Offset in .debug_info:
; CHECK-NEXT: Size:
; CHECK-NEXT: Offset     Linkage  Kind     Name
; CHECK-DAG:  0x0000003a EXTERNAL TYPE     "C"
; CHECK-DAG:  0x0000002e STATIC   TYPE     "int"

%struct.C = type { i8 }

@_ZN1C22static_member_variableE = global i32 0, align 4
@global_variable = global %struct.C zeroinitializer, align 1
@_ZN2ns25global_namespace_variableE = global i32 1, align 4

; Function Attrs: nounwind uwtable
define void @_ZN1C15member_functionEv(%struct.C* %this) #0 align 2 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.C** %this.addr}, metadata !31), !dbg !33
  %this1 = load %struct.C** %this.addr
  store i32 0, i32* @_ZN1C22static_member_variableE, align 4, !dbg !34
  ret void, !dbg !34
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @_ZN1C22static_member_functionEv() #0 align 2 {
entry:
  %0 = load i32* @_ZN1C22static_member_variableE, align 4, !dbg !35
  ret i32 %0, !dbg !35
}

; Function Attrs: nounwind uwtable
define i32 @_Z15global_functionv() #0 {
entry:
  ret i32 -1, !dbg !36
}

; Function Attrs: nounwind uwtable
define void @_ZN2ns25global_namespace_functionEv() #0 {
entry:
  call void @_ZN1C15member_functionEv(%struct.C* @global_variable), !dbg !37
  ret void, !dbg !37
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!30}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 (trunk 191222) (llvm/trunk 191239)", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !17, metadata !26, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/pubnames.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"pubnames.cpp", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"C", i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !5, i32 0, null, null, metadata !"_ZTS1C"} ; [ DW_TAG_structure_type ] [C] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6, metadata !8, metadata !13}
!6 = metadata !{i32 786445, metadata !1, metadata !4, metadata !"static_member_variable", i32 4, i64 0, i64 0, i64 0, i32 4096, metadata !7, null} ; [ DW_TAG_member ] [static_member_variable] [line 4, size 0, align 0, offset 0] [static] [from int]
!7 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = metadata !{i32 786478, metadata !1, metadata !4, metadata !"member_function", metadata !"member_function", metadata !"_ZN1C15member_functionEv", i32 2, metadata !9, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !12, i32 2} ; [ DW_TAG_subprogram ] [line 2] [member_function]
!9 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{null, metadata !11}
!11 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !4} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from C]
!12 = metadata !{i32 786468}
!13 = metadata !{i32 786478, metadata !1, metadata !4, metadata !"static_member_function", metadata !"static_member_function", metadata !"_ZN1C22static_member_functionEv", i32 3, metadata !14, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !16, i32 3} ; [ DW_TAG_subprogram ] [line 3] [static_member_function]
!14 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !15, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = metadata !{metadata !7}
!16 = metadata !{i32 786468}
!17 = metadata !{metadata !18, metadata !19, metadata !20, metadata !22}
!18 = metadata !{i32 786478, metadata !1, null, metadata !"member_function", metadata !"member_function", metadata !"_ZN1C15member_functionEv", i32 9, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%struct.C*)* @_ZN1C15member_functionEv, null, metadata !8, metadata !2, i32 9} ; [ DW_TAG_subprogram ] [line 9] [def] [member_function]
!19 = metadata !{i32 786478, metadata !1, null, metadata !"static_member_function", metadata !"static_member_function", metadata !"_ZN1C22static_member_functionEv", i32 11, metadata !14, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_ZN1C22static_member_functionEv, null, metadata !13, metadata !2, i32 11} ; [ DW_TAG_subprogram ] [line 11] [def] [static_member_function]
!20 = metadata !{i32 786478, metadata !1, metadata !21, metadata !"global_function", metadata !"global_function", metadata !"_Z15global_functionv", i32 15, metadata !14, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z15global_functionv, null, null, metadata !2, i32 15} ; [ DW_TAG_subprogram ] [line 15] [def] [global_function]
!21 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/pubnames.cpp]
!22 = metadata !{i32 786478, metadata !1, metadata !23, metadata !"global_namespace_function", metadata !"global_namespace_function", metadata !"_ZN2ns25global_namespace_functionEv", i32 18, metadata !24, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_ZN2ns25global_namespace_functionEv, null, null, metadata !2, i32 18} ; [ DW_TAG_subprogram ] [line 18] [def] [global_namespace_function]
!23 = metadata !{i32 786489, metadata !1, null, metadata !"ns", i32 17} ; [ DW_TAG_namespace ] [ns] [line 17]
!24 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !25, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!25 = metadata !{null}
!26 = metadata !{metadata !27, metadata !28, metadata !29}
!27 = metadata !{i32 786484, i32 0, metadata !4, metadata !"static_member_variable", metadata !"static_member_variable", metadata !"_ZN1C22static_member_variableE", metadata !21, i32 7, metadata !7, i32 0, i32 1, i32* @_ZN1C22static_member_variableE, metadata !6} ; [ DW_TAG_variable ] [static_member_variable] [line 7] [def]
!28 = metadata !{i32 786484, i32 0, null, metadata !"global_variable", metadata !"global_variable", metadata !"", metadata !21, i32 13, metadata !4, i32 0, i32 1, %struct.C* @global_variable, null} ; [ DW_TAG_variable ] [global_variable] [line 13] [def]
!29 = metadata !{i32 786484, i32 0, metadata !23, metadata !"global_namespace_variable", metadata !"global_namespace_variable", metadata !"_ZN2ns25global_namespace_variableE", metadata !21, i32 19, metadata !7, i32 0, i32 1, i32* @_ZN2ns25global_namespace_variableE, null} ; [ DW_TAG_variable ] [global_namespace_variable] [line 19] [def]
!30 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!31 = metadata !{i32 786689, metadata !18, metadata !"this", null, i32 16777216, metadata !32, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!32 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !4} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from C]
!33 = metadata !{i32 0, i32 0, metadata !18, null}
!34 = metadata !{i32 9, i32 0, metadata !18, null}
!35 = metadata !{i32 11, i32 0, metadata !19, null}
!36 = metadata !{i32 15, i32 0, metadata !20, null}
!37 = metadata !{i32 18, i32 0, metadata !22, null}
