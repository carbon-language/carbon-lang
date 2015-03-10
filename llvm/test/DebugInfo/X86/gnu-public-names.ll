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

; CHECK: [[STATIC_MEM_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_specification {{.*}} "static_member_variable"

; CHECK: [[C:0x[0-9a-f]+]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}} "C"

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name {{.*}} "static_member_variable"

; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "member_function"

; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "static_member_function"

; CHECK: [[INT:0x[0-9a-f]+]]: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name {{.*}} "int"

; CHECK: [[GLOB_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "global_variable"

; CHECK: [[NS:0x[0-9a-f]+]]: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name {{.*}} "ns"

; CHECK: [[GLOB_NS_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "global_namespace_variable"
; CHECK-NOT: DW_AT_specification
; CHECK: DW_AT_location
; CHECK-NOT: DW_AT_specification

; CHECK: [[D_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "d"
; CHECK-NOT: DW_AT_specification
; CHECK: DW_AT_location
; CHECK-NOT: DW_AT_specification

; CHECK: [[D:0x[0-9a-f]+]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}} "D"

; CHECK: [[GLOB_NS_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "global_namespace_function"

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
; CHECK: [[OUTER_ANON_C:.*]]: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "c"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     NULL
; CHECK-NOT: {{DW_TAG|NULL}}
; FIXME: We probably shouldn't bother describing the implicit
; import of the preceding anonymous namespace. This should be fixed
; in clang.
; CHECK:     DW_TAG_imported_module
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   NULL
; CHECK-NOT: {{DW_TAG|NULL}}

; CHECK: [[ANON:.*]]: DW_TAG_namespace
; CHECK-NOT:   DW_AT_name
; CHECK: [[ANON_INNER:.*]]:  DW_TAG_namespace
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "inner"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[ANON_INNER_B:.*]]: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "b"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     NULL
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[ANON_I:.*]]: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "i"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   NULL
; CHECK-NOT: {{DW_TAG|NULL}}

; CHECK: [[MEM_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_specification {{.*}} "_ZN1C15member_functionEv"

; CHECK: [[STATIC_MEM_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_specification {{.*}} "_ZN1C22static_member_functionEv"

; CHECK: [[GLOBAL_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_linkage_name
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
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !50, metadata !MDExpression()), !dbg !52
  %this1 = load %struct.C*, %struct.C** %this.addr
  store i32 0, i32* @_ZN1C22static_member_variableE, align 4, !dbg !53
  ret void, !dbg !54
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @_ZN1C22static_member_functionEv() #0 align 2 {
entry:
  %0 = load i32, i32* @_ZN1C22static_member_variableE, align 4, !dbg !55
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
  %0 = load i32, i32* @_ZN12_GLOBAL__N_11iE, align 4, !dbg !60
  %call = call i32* @_Z2f3v(), !dbg !60
  %1 = load i32, i32* %call, align 4, !dbg !60
  %add = add nsw i32 %0, %1, !dbg !60
  %2 = load i32, i32* @_ZN12_GLOBAL__N_15inner1bE, align 4, !dbg !60
  %add1 = add nsw i32 %add, %2, !dbg !60
  %3 = load i32, i32* @_ZN5outer12_GLOBAL__N_11cE, align 4, !dbg !60
  %add2 = add nsw i32 %add1, %3, !dbg !60
  ret i32 %add2, !dbg !60
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!47, !48}
!llvm.ident = !{!49}

!0 = !MDCompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !3, subprograms: !19, globals: !32, imports: !45)
!1 = !MDFile(filename: "pubnames.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4, !15}
!4 = !MDCompositeType(tag: DW_TAG_structure_type, name: "C", line: 1, size: 8, align: 8, file: !1, elements: !5, identifier: "_ZTS1C")
!5 = !{!6, !8, !12}
!6 = !MDDerivedType(tag: DW_TAG_member, name: "static_member_variable", line: 4, flags: DIFlagStaticMember, file: !1, scope: !"_ZTS1C", baseType: !7)
!7 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !MDSubprogram(name: "member_function", linkageName: "_ZN1C15member_functionEv", line: 2, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !1, scope: !"_ZTS1C", type: !9)
!9 = !MDSubroutineType(types: !10)
!10 = !{null, !11}
!11 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !"_ZTS1C")
!12 = !MDSubprogram(name: "static_member_function", linkageName: "_ZN1C22static_member_functionEv", line: 3, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !1, scope: !"_ZTS1C", type: !13)
!13 = !MDSubroutineType(types: !14)
!14 = !{!7}
!15 = !MDCompositeType(tag: DW_TAG_structure_type, name: "D", line: 28, size: 32, align: 32, file: !1, scope: !16, elements: !17, identifier: "_ZTSN2ns1DE")
!16 = !MDNamespace(name: "ns", line: 23, file: !1, scope: null)
!17 = !{!18}
!18 = !MDDerivedType(tag: DW_TAG_member, name: "A", line: 29, size: 32, align: 32, file: !1, scope: !"_ZTSN2ns1DE", baseType: !7)
!19 = !{!20, !21, !22, !24, !27, !31}
!20 = !MDSubprogram(name: "member_function", linkageName: "_ZN1C15member_functionEv", line: 9, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 9, file: !1, scope: !"_ZTS1C", type: !9, function: void (%struct.C*)* @_ZN1C15member_functionEv, declaration: !8, variables: !2)
!21 = !MDSubprogram(name: "static_member_function", linkageName: "_ZN1C22static_member_functionEv", line: 13, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 13, file: !1, scope: !"_ZTS1C", type: !13, function: i32 ()* @_ZN1C22static_member_functionEv, declaration: !12, variables: !2)
!22 = !MDSubprogram(name: "global_function", linkageName: "_Z15global_functionv", line: 19, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 19, file: !1, scope: !23, type: !13, function: i32 ()* @_Z15global_functionv, variables: !2)
!23 = !MDFile(filename: "pubnames.cpp", directory: "/tmp/dbginfo")
!24 = !MDSubprogram(name: "global_namespace_function", linkageName: "_ZN2ns25global_namespace_functionEv", line: 24, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 24, file: !1, scope: !16, type: !25, function: void ()* @_ZN2ns25global_namespace_functionEv, variables: !2)
!25 = !MDSubroutineType(types: !26)
!26 = !{null}
!27 = !MDSubprogram(name: "f3", linkageName: "_Z2f3v", line: 37, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 37, file: !1, scope: !23, type: !28, function: i32* ()* @_Z2f3v, variables: !2)
!28 = !MDSubroutineType(types: !29)
!29 = !{!30}
!30 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !7)
!31 = !MDSubprogram(name: "f7", linkageName: "_Z2f7v", line: 54, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 54, file: !1, scope: !23, type: !13, function: i32 ()* @_Z2f7v, variables: !2)
!32 = !{!33, !34, !35, !36, !37, !38, !41, !44}
!33 = !MDGlobalVariable(name: "static_member_variable", linkageName: "_ZN1C22static_member_variableE", line: 7, isLocal: false, isDefinition: true, scope: null, file: !23, type: !7, variable: i32* @_ZN1C22static_member_variableE, declaration: !6)
!34 = !MDGlobalVariable(name: "global_variable", line: 17, isLocal: false, isDefinition: true, scope: null, file: !23, type: !"_ZTS1C", variable: %struct.C* @global_variable)
!35 = !MDGlobalVariable(name: "global_namespace_variable", linkageName: "_ZN2ns25global_namespace_variableE", line: 27, isLocal: false, isDefinition: true, scope: !16, file: !23, type: !7, variable: i32* @_ZN2ns25global_namespace_variableE)
!36 = !MDGlobalVariable(name: "d", linkageName: "_ZN2ns1dE", line: 30, isLocal: false, isDefinition: true, scope: !16, file: !23, type: !"_ZTSN2ns1DE", variable: %"struct.ns::D"* @_ZN2ns1dE)
!37 = !MDGlobalVariable(name: "z", line: 38, isLocal: true, isDefinition: true, scope: !27, file: !23, type: !7, variable: i32* @_ZZ2f3vE1z)
!38 = !MDGlobalVariable(name: "c", linkageName: "_ZN5outer12_GLOBAL__N_11cE", line: 50, isLocal: true, isDefinition: true, scope: !39, file: !23, type: !7, variable: i32* @_ZN5outer12_GLOBAL__N_11cE)
!39 = !MDNamespace(line: 49, file: !1, scope: !40)
!40 = !MDNamespace(name: "outer", line: 48, file: !1, scope: null)
!41 = !MDGlobalVariable(name: "b", linkageName: "_ZN12_GLOBAL__N_15inner1bE", line: 44, isLocal: true, isDefinition: true, scope: !42, file: !23, type: !7, variable: i32* @_ZN12_GLOBAL__N_15inner1bE)
!42 = !MDNamespace(name: "inner", line: 43, file: !1, scope: !43)
!43 = !MDNamespace(line: 33, file: !1, scope: null)
!44 = !MDGlobalVariable(name: "i", linkageName: "_ZN12_GLOBAL__N_11iE", line: 34, isLocal: true, isDefinition: true, scope: !43, file: !23, type: !7, variable: i32* @_ZN12_GLOBAL__N_11iE)
!45 = !{!46}
!46 = !MDImportedEntity(tag: DW_TAG_imported_module, line: 40, scope: !40, entity: !39)
!47 = !{i32 2, !"Dwarf Version", i32 4}
!48 = !{i32 2, !"Debug Info Version", i32 3}
!49 = !{!"clang version 3.5.0 "}
!50 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !20, type: !51)
!51 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !"_ZTS1C")
!52 = !MDLocation(line: 0, scope: !20)
!53 = !MDLocation(line: 10, scope: !20)
!54 = !MDLocation(line: 11, scope: !20)
!55 = !MDLocation(line: 14, scope: !21)
!56 = !MDLocation(line: 20, scope: !22)
!57 = !MDLocation(line: 25, scope: !24)
!58 = !MDLocation(line: 26, scope: !24)
!59 = !MDLocation(line: 39, scope: !27)
!60 = !MDLocation(line: 55, scope: !31)
