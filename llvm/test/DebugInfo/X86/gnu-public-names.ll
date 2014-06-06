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
;   struct D {
;     int A;
;   } d;
; }

; ASM: .section        .debug_gnu_pubnames
; ASM: .byte   32                      # Kind: VARIABLE, EXTERNAL
; ASM-NEXT: .asciz  "global_variable"       # External Name

; ASM: .section        .debug_gnu_pubtypes
; ASM: .byte   16                      # Kind: TYPE, EXTERNAL
; ASM-NEXT: .asciz  "C"                     # External Name

; CHECK: .debug_info contents:
; CHECK: Compile Unit:
; CHECK: DW_AT_GNU_pubnames [DW_FORM_flag_present]   (true)
; CHECK-NOT: DW_AT_GNU_pubtypes [

; CHECK: [[C:0x[0-9a-f]+]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}} "C"

; CHECK: [[STATIC_MEM_DECL:0x[0-9a-f]+]]: DW_TAG_member
; CHECK-NEXT: DW_AT_name {{.*}} "static_member_variable"

; CHECK: [[MEM_FUNC_DECL:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "member_function"

; CHECK: [[STATIC_MEM_FUNC_DECL:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "static_member_function"

; CHECK: [[INT:0x[0-9a-f]+]]: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name {{.*}} "int"

; CHECK: [[STATIC_MEM_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_specification {{.*}} {[[STATIC_MEM_DECL]]}

; CHECK: [[GLOB_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "global_variable"

; CHECK: [[NS:0x[0-9a-f]+]]: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name {{.*}} "ns"

; CHECK: [[GLOB_NS_VAR_DECL:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "global_namespace_variable"

; CHECK: [[D_VAR_DECL:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "d"

; CHECK: [[D:0x[0-9a-f]+]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}} "D"

; CHECK: [[GLOB_NS_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "global_namespace_function"

; CHECK: [[GLOB_NS_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_specification {{.*}} {[[GLOB_NS_VAR_DECL]]}

; CHECK: [[D_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_specification {{.*}} {[[D_VAR_DECL]]}

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "f3"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[F3_Z:.*]]:   DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "z"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_AT_location
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   NULL
; CHECK-NOT: {{DW_TAG|NULL}}

; CHECK: [[OUTER:.*]]: DW_TAG_namespace
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "outer"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[OUTER_ANON:.*]]:  DW_TAG_namespace
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK-NOT:     DW_AT_name
; CHECK: [[OUTER_ANON_C_DECL:.*]]:     DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "c"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     NULL
; CHECK-NOT: {{DW_TAG|NULL}}
; FIXME: We probably shouldn't bother describing the implicit
; import of the preceeding anonymous namespace. This should be fixed
; in clang.
; CHECK:     DW_TAG_imported_module
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   NULL
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[OUTER_ANON_C:.*]]: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK-NEXT:   DW_AT_specification {{.*}} {[[OUTER_ANON_C_DECL]]}

; CHECK: [[ANON:.*]]: DW_TAG_namespace
; CHECK-NOT:   DW_AT_name
; CHECK: [[ANON_INNER:.*]]:  DW_TAG_namespace
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "inner"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[ANON_INNER_B_DECL:.*]]:     DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "b"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     NULL
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[ANON_I_DECL:.*]]:   DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "i"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   NULL
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[ANON_INNER_B:.*]]: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK-NEXT:   DW_AT_specification {{.*}} {[[ANON_INNER_B_DECL]]}
; CHECK: [[ANON_I:.*]]: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK-NEXT:   DW_AT_specification {{.*}} {[[ANON_I_DECL]]}

; CHECK: [[MEM_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_specification {{.*}} {[[MEM_FUNC_DECL]]}

; CHECK: [[STATIC_MEM_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_specification {{.*}} {[[STATIC_MEM_FUNC_DECL]]}

; CHECK: [[GLOBAL_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "global_function"

; CHECK-LABEL: .debug_gnu_pubnames contents:
; CHECK-NEXT: length = {{.*}} version = 0x0002 unit_offset = 0x00000000 unit_size = {{.*}}
; CHECK-NEXT: Offset     Linkage  Kind     Name
; CHECK-DAG:  [[GLOBAL_FUNC]] EXTERNAL FUNCTION "global_function"
; CHECK-DAG:  [[NS]] EXTERNAL TYPE     "ns"
; CHECK-DAG:  [[MEM_FUNC]] EXTERNAL FUNCTION "C::member_function"
; CHECK-DAG:  [[GLOB_VAR]] EXTERNAL VARIABLE "global_variable"
; CHECK-DAG:  [[GLOB_NS_VAR]] EXTERNAL VARIABLE "ns::global_namespace_variable"
; CHECK-DAG:  [[GLOB_NS_FUNC]] EXTERNAL FUNCTION "ns::global_namespace_function"
; CHECK-DAG:  [[D_VAR]] EXTERNAL VARIABLE "ns::d"
; CHECK-DAG:  [[STATIC_MEM_VAR]] EXTERNAL VARIABLE "C::static_member_variable"
; CHECK-DAG:  [[STATIC_MEM_FUNC]] EXTERNAL FUNCTION "C::static_member_function"
; CHECK-DAG:  [[ANON]] EXTERNAL TYPE "(anonymous namespace)"
; CHECK-DAG:  [[ANON_INNER]] EXTERNAL TYPE "(anonymous namespace)::inner"
; CHECK-DAG:  [[OUTER]] EXTERNAL TYPE "outer"
; CHECK-DAG:  [[OUTER_ANON]] EXTERNAL TYPE "outer::(anonymous namespace)"
; CHECK-DAG:  [[ANON_I]] STATIC VARIABLE "(anonymous namespace)::i"
; CHECK-DAG:  [[ANON_INNER_B]] STATIC VARIABLE "(anonymous namespace)::inner::b"
; CHECK-DAG:  [[OUTER_ANON_C]] STATIC VARIABLE "outer::(anonymous namespace)::c"

; GCC Doesn't put local statics in pubnames, but it seems not unreasonable and
; comes out naturally from LLVM's implementation, so I'm OK with it for now. If
; it's demonstrated that this is a major size concern or degrades debug info
; consumer behavior, feel free to change it.

; CHECK-DAG:  [[F3_Z]] STATIC VARIABLE "f3::z"


; CHECK-LABEL: debug_gnu_pubtypes contents:
; CHECK: Offset     Linkage  Kind     Name
; CHECK-DAG:  [[C]] EXTERNAL TYPE     "C"
; CHECK-DAG:  [[D]] EXTERNAL TYPE     "ns::D"
; CHECK-DAG:  [[INT]] STATIC   TYPE     "int"

%struct.C = type { i8 }
%"struct.ns::D" = type { i32 }

@_ZN1C22static_member_variableE = global i32 0, align 4
@global_variable = global %struct.C zeroinitializer, align 1
@_ZN2ns25global_namespace_variableE = global i32 1, align 4
@_ZN2ns1dE = global %"struct.ns::D" zeroinitializer, align 4
@_ZZ2f3vE1z = internal global i32 0, align 4
@_ZN12_GLOBAL__N_11iE = internal global i32 0, align 4
@_ZN12_GLOBAL__N_15inner1bE = internal global i32 0, align 4
@_ZN5outer12_GLOBAL__N_11cE = internal global i32 0, align 4

; Function Attrs: nounwind uwtable
define void @_ZN1C15member_functionEv(%struct.C* %this) #0 align 2 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.C** %this.addr}, metadata !50), !dbg !52
  %this1 = load %struct.C** %this.addr
  store i32 0, i32* @_ZN1C22static_member_variableE, align 4, !dbg !53
  ret void, !dbg !54
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @_ZN1C22static_member_functionEv() #0 align 2 {
entry:
  %0 = load i32* @_ZN1C22static_member_variableE, align 4, !dbg !55
  ret i32 %0, !dbg !55
}

; Function Attrs: nounwind uwtable
define i32 @_Z15global_functionv() #0 {
entry:
  ret i32 -1, !dbg !56
}

; Function Attrs: nounwind uwtable
define void @_ZN2ns25global_namespace_functionEv() #0 {
entry:
  call void @_ZN1C15member_functionEv(%struct.C* @global_variable), !dbg !57
  ret void, !dbg !58
}

; Function Attrs: nounwind uwtable
define i32* @_Z2f3v() #0 {
entry:
  ret i32* @_ZZ2f3vE1z, !dbg !59
}

; Function Attrs: nounwind uwtable
define i32 @_Z2f7v() #0 {
entry:
  %0 = load i32* @_ZN12_GLOBAL__N_11iE, align 4, !dbg !60
  %call = call i32* @_Z2f3v(), !dbg !60
  %1 = load i32* %call, align 4, !dbg !60
  %add = add nsw i32 %0, %1, !dbg !60
  %2 = load i32* @_ZN12_GLOBAL__N_15inner1bE, align 4, !dbg !60
  %add1 = add nsw i32 %add, %2, !dbg !60
  %3 = load i32* @_ZN5outer12_GLOBAL__N_11cE, align 4, !dbg !60
  %add2 = add nsw i32 %add1, %3, !dbg !60
  ret i32 %add2, !dbg !60
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!47, !48}
!llvm.ident = !{!49}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !19, metadata !32, metadata !45, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/pubnames.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"pubnames.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !15}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"C", i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !5, i32 0, null, null, metadata !"_ZTS1C"} ; [ DW_TAG_structure_type ] [C] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6, metadata !8, metadata !12}
!6 = metadata !{i32 786445, metadata !1, metadata !"_ZTS1C", metadata !"static_member_variable", i32 4, i64 0, i64 0, i64 0, i32 4096, metadata !7, null} ; [ DW_TAG_member ] [static_member_variable] [line 4, size 0, align 0, offset 0] [static] [from int]
!7 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1C", metadata !"member_function", metadata !"member_function", metadata !"_ZN1C15member_functionEv", i32 2, metadata !9, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, null, i32 2} ; [ DW_TAG_subprogram ] [line 2] [member_function]
!9 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{null, metadata !11}
!11 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1C]
!12 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1C", metadata !"static_member_function", metadata !"static_member_function", metadata !"_ZN1C22static_member_functionEv", i32 3, metadata !13, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, null, i32 3} ; [ DW_TAG_subprogram ] [line 3] [static_member_function]
!13 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !14, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = metadata !{metadata !7}
!15 = metadata !{i32 786451, metadata !1, metadata !16, metadata !"D", i32 28, i64 32, i64 32, i32 0, i32 0, null, metadata !17, i32 0, null, null, metadata !"_ZTSN2ns1DE"} ; [ DW_TAG_structure_type ] [D] [line 28, size 32, align 32, offset 0] [def] [from ]
!16 = metadata !{i32 786489, metadata !1, null, metadata !"ns", i32 23} ; [ DW_TAG_namespace ] [ns] [line 23]
!17 = metadata !{metadata !18}
!18 = metadata !{i32 786445, metadata !1, metadata !"_ZTSN2ns1DE", metadata !"A", i32 29, i64 32, i64 32, i64 0, i32 0, metadata !7} ; [ DW_TAG_member ] [A] [line 29, size 32, align 32, offset 0] [from int]
!19 = metadata !{metadata !20, metadata !21, metadata !22, metadata !24, metadata !27, metadata !31}
!20 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1C", metadata !"member_function", metadata !"member_function", metadata !"_ZN1C15member_functionEv", i32 9, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%struct.C*)* @_ZN1C15member_functionEv, null, metadata !8, metadata !2, i32 9} ; [ DW_TAG_subprogram ] [line 9] [def] [member_function]
!21 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1C", metadata !"static_member_function", metadata !"static_member_function", metadata !"_ZN1C22static_member_functionEv", i32 13, metadata !13, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_ZN1C22static_member_functionEv, null, metadata !12, metadata !2, i32 13} ; [ DW_TAG_subprogram ] [line 13] [def] [static_member_function]
!22 = metadata !{i32 786478, metadata !1, metadata !23, metadata !"global_function", metadata !"global_function", metadata !"_Z15global_functionv", i32 19, metadata !13, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z15global_functionv, null, null, metadata !2, i32 19} ; [ DW_TAG_subprogram ] [line 19] [def] [global_function]
!23 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/pubnames.cpp]
!24 = metadata !{i32 786478, metadata !1, metadata !16, metadata !"global_namespace_function", metadata !"global_namespace_function", metadata !"_ZN2ns25global_namespace_functionEv", i32 24, metadata !25, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_ZN2ns25global_namespace_functionEv, null, null, metadata !2, i32 24} ; [ DW_TAG_subprogram ] [line 24] [def] [global_namespace_function]
!25 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !26, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!26 = metadata !{null}
!27 = metadata !{i32 786478, metadata !1, metadata !23, metadata !"f3", metadata !"f3", metadata !"_Z2f3v", i32 37, metadata !28, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32* ()* @_Z2f3v, null, null, metadata !2, i32 37} ; [ DW_TAG_subprogram ] [line 37] [def] [f3]
!28 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !29, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!29 = metadata !{metadata !30}
!30 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !7} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!31 = metadata !{i32 786478, metadata !1, metadata !23, metadata !"f7", metadata !"f7", metadata !"_Z2f7v", i32 54, metadata !13, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z2f7v, null, null, metadata !2, i32 54} ; [ DW_TAG_subprogram ] [line 54] [def] [f7]
!32 = metadata !{metadata !33, metadata !34, metadata !35, metadata !36, metadata !37, metadata !38, metadata !41, metadata !44}
!33 = metadata !{i32 786484, i32 0, metadata !4, metadata !"static_member_variable", metadata !"static_member_variable", metadata !"_ZN1C22static_member_variableE", metadata !23, i32 7, metadata !7, i32 0, i32 1, i32* @_ZN1C22static_member_variableE, metadata !6} ; [ DW_TAG_variable ] [static_member_variable] [line 7] [def]
!34 = metadata !{i32 786484, i32 0, null, metadata !"global_variable", metadata !"global_variable", metadata !"", metadata !23, i32 17, metadata !"_ZTS1C", i32 0, i32 1, %struct.C* @global_variable, null} ; [ DW_TAG_variable ] [global_variable] [line 17] [def]
!35 = metadata !{i32 786484, i32 0, metadata !16, metadata !"global_namespace_variable", metadata !"global_namespace_variable", metadata !"_ZN2ns25global_namespace_variableE", metadata !23, i32 27, metadata !7, i32 0, i32 1, i32* @_ZN2ns25global_namespace_variableE, null} ; [ DW_TAG_variable ] [global_namespace_variable] [line 27] [def]
!36 = metadata !{i32 786484, i32 0, metadata !16, metadata !"d", metadata !"d", metadata !"_ZN2ns1dE", metadata !23, i32 30, metadata !"_ZTSN2ns1DE", i32 0, i32 1, %"struct.ns::D"* @_ZN2ns1dE, null} ; [ DW_TAG_variable ] [d] [line 30] [def]
!37 = metadata !{i32 786484, i32 0, metadata !27, metadata !"z", metadata !"z", metadata !"", metadata !23, i32 38, metadata !7, i32 1, i32 1, i32* @_ZZ2f3vE1z, null} ; [ DW_TAG_variable ] [z] [line 38] [local] [def]
!38 = metadata !{i32 786484, i32 0, metadata !39, metadata !"c", metadata !"c", metadata !"_ZN5outer12_GLOBAL__N_11cE", metadata !23, i32 50, metadata !7, i32 1, i32 1, i32* @_ZN5outer12_GLOBAL__N_11cE, null} ; [ DW_TAG_variable ] [c] [line 50] [local] [def]
!39 = metadata !{i32 786489, metadata !1, metadata !40, metadata !"", i32 49} ; [ DW_TAG_namespace ] [line 49]
!40 = metadata !{i32 786489, metadata !1, null, metadata !"outer", i32 48} ; [ DW_TAG_namespace ] [outer] [line 48]
!41 = metadata !{i32 786484, i32 0, metadata !42, metadata !"b", metadata !"b", metadata !"_ZN12_GLOBAL__N_15inner1bE", metadata !23, i32 44, metadata !7, i32 1, i32 1, i32* @_ZN12_GLOBAL__N_15inner1bE, null} ; [ DW_TAG_variable ] [b] [line 44] [local] [def]
!42 = metadata !{i32 786489, metadata !1, metadata !43, metadata !"inner", i32 43} ; [ DW_TAG_namespace ] [inner] [line 43]
!43 = metadata !{i32 786489, metadata !1, null, metadata !"", i32 33} ; [ DW_TAG_namespace ] [line 33]
!44 = metadata !{i32 786484, i32 0, metadata !43, metadata !"i", metadata !"i", metadata !"_ZN12_GLOBAL__N_11iE", metadata !23, i32 34, metadata !7, i32 1, i32 1, i32* @_ZN12_GLOBAL__N_11iE, null} ; [ DW_TAG_variable ] [i] [line 34] [local] [def]
!45 = metadata !{metadata !46}
!46 = metadata !{i32 786490, metadata !40, metadata !39, i32 40} ; [ DW_TAG_imported_module ]
!47 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!48 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!49 = metadata !{metadata !"clang version 3.5.0 "}
!50 = metadata !{i32 786689, metadata !20, metadata !"this", null, i32 16777216, metadata !51, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!51 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1C]
!52 = metadata !{i32 0, i32 0, metadata !20, null}
!53 = metadata !{i32 10, i32 0, metadata !20, null}
!54 = metadata !{i32 11, i32 0, metadata !20, null}
!55 = metadata !{i32 14, i32 0, metadata !21, null}
!56 = metadata !{i32 20, i32 0, metadata !22, null}
!57 = metadata !{i32 25, i32 0, metadata !24, null}
!58 = metadata !{i32 26, i32 0, metadata !24, null}
!59 = metadata !{i32 39, i32 0, metadata !27, null}
!60 = metadata !{i32 55, i32 0, metadata !31, null}
