; REQUIRES: object-emission

; RUN: llc < %s -o %t -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu
; RUN: llvm-dwarfdump %t | FileCheck --check-prefix=CHECK --check-prefix=SINGLE %s
; RUN: llvm-readobj -s -t %t | FileCheck --check-prefix=OBJ_COMMON %s

; RUN: llc < %s -split-dwarf=Enable -o %t -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu
; RUN: llvm-dwarfdump %t | FileCheck --check-prefix=CHECK --check-prefix=FISSION %s
; RUN: llvm-readobj -s -t %t | FileCheck --check-prefix=OBJ_COMMON --check-prefix=OBJ_FISSION %s

; Generated from bar.cpp:

; #line 1 "bar.h"
; struct bar {};
; #line 2 "bar.cpp"

; struct bar b;

; void foo(void) {
;   struct baz {};
;   baz b;
; }

; namespace echidna {
; namespace capybara {
; namespace mongoose {
; class fluffy {
;   int a;
;   int b;
; };

; fluffy animal;
; }
; }
; }

; namespace {
; struct walrus {
;   walrus() {}
; };
; }

; walrus w;

; struct wombat {
;   struct {
;     int a;
;     int b;
;   } a_b;
; };

; wombat wom;

; SINGLE-LABEL: .debug_info contents:
; FISSION-LABEL: .debug_info.dwo contents:
; CHECK: Compile Unit: length = [[CU_SIZE:[0-9a-f]+]]

; CHECK: [[BAR:^0x........]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_declaration
; CHECK-NEXT: DW_AT_signature {{.*}} (0x1d02f3be30cc5688)
; CHECK: [[FLUFFY:^0x........]]: DW_TAG_class_type
; CHECK-NEXT: DW_AT_declaration
; CHECK-NEXT: DW_AT_signature {{.*}} (0xb04af47397402e77)

; Ensure the CU-local type 'walrus' is not placed in a type unit.
; CHECK: [[WALRUS:^0x........]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}"walrus"
; CHECK-NEXT: DW_AT_byte_size
; CHECK-NEXT: DW_AT_decl_file
; CHECK-NEXT: DW_AT_decl_line


; CHECK: [[WOMBAT:^0x........]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_declaration
; CHECK-NEXT: DW_AT_signature {{.*}} (0xfd756cee88f8a118)

; FISSION-LABEL: .debug_types contents:
; FISSION-NOT: type_signature
; FISSION-LABEL: type_signature = 0x1d02f3be30cc5688
; FISSION: DW_TAG_type_unit
; FISSION-NEXT: DW_AT_GNU_dwo_name{{.*}}"bar.dwo"
; FISSION-NEXT: DW_AT_comp_dir{{.*}}"/tmp/dbginfo"
; FISSION-NOT: type_signature
; FISSION-LABEL: type_signature = 0xb04af47397402e77
; FISSION-NOT: type_signature
; FISSION-LABEL: type_signature = 0xfd756cee88f8a118
; FISSION-NOT: type_signature
; FISSION-LABEL: type_signature = 0xe94f6d3843e62d6b

; SINGLE-LABEL: .debug_types contents:
; FISSION-LABEL: .debug_types.dwo contents:

; Check that we generate a hash for bar and the value.
; CHECK-NOT: type_signature
; CHECK-LABEL: type_signature = 0x1d02f3be30cc5688
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}"bar"


; Check that we generate a hash for fluffy and the value.
; CHECK-NOT: type_signature
; CHECK-LABEL: type_signature = 0xb04af47397402e77
; CHECK-NOT: DW_AT_GNU_odr_signature [DW_FORM_data8]   (0x9a0124d5a0c21c52)
; CHECK: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}}"echidna"
; CHECK: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}}"capybara"
; CHECK: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}}"mongoose"
; CHECK: DW_TAG_class_type
; CHECK-NEXT: DW_AT_name{{.*}}"fluffy"

; Check that we generate a hash for wombat and the value, but not for the
; anonymous type contained within.
; CHECK-NOT: type_signature
; CHECK-LABEL: type_signature = 0xfd756cee88f8a118
; CHECK-NOT: DW_AT_GNU_odr_signature [DW_FORM_data8] (0x685bcc220141e9d7)
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}"wombat"

; CHECK-NOT: type_signature
; CHECK-LABEL: type_signature = 0xe94f6d3843e62d6b
; CHECK: DW_TAG_type_unit
; CHECK: DW_AT_stmt_list [DW_FORM_sec_offset] (0x00000000)
; CHECK-NOT: NULL
; CHECK-NOT: DW_AT_GNU_odr_signature
; CHECK: DW_TAG_structure_type
; The signature for the outer 'wombat' type
; CHECK: DW_AT_signature [DW_FORM_ref_sig8] (0xfd756cee88f8a118)
; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_AT_name
; CHECK-NOT: DW_AT_GNU_odr_signature
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"a"

; CHECK-LABEL: .debug_line contents:
; CHECK: Line table prologue
; CHECK-NOT: file_names[
; SINGLE: file_names{{.*}} bar.h
; CHECK: file_names{{.*}} bar.cpp
; CHECK-NOT: file_names[

; CHECK-LABEL: .debug_line.dwo contents:
; FISSION: Line table prologue
; FISSION: opcode_base: 1
; FISSION-NOT: standard_opcode_lengths
; FISSION-NOT: include_directories
; FISSION-NOT: file_names[
; FISSION: file_names{{.*}} bar.h
; FISSION: file_names{{.*}} bar.cpp
; FISSION-NOT: file_names[

; CHECK-LABEL: .debug_str contents:

; Use the unit size as a rough hash/identifier for the unit we're dealing with
; it happens to be unambiguous at the moment, but it's hardly ideal.
; CHECK-LABEL: .debug_pubtypes contents:
; Don't emit pubtype entries for type DIEs in the compile unit that just indirect to a type unit.
; CHECK-NEXT: unit_size = [[CU_SIZE]]
; CHECK-NEXT: Offset Name
; CHECK-DAG: [[BAR]] "bar"
; CHECK-DAG: [[WALRUS]] "(anonymous namespace)::walrus"
; CHECK-DAG: [[WOMBAT]] "wombat"
; CHECK-DAG: [[FLUFFY]] "echidna::capybara::mongoose::fluffy"

; Make sure debug_types are in comdat groups. This could be more rigid to check
; that they're the right comdat groups (each type in a separate comdat group,
; etc)
; OBJ_COMMON: Name: .debug_types (
; OBJ_COMMON-NOT: }
; OBJ_COMMON: SHF_GROUP

; Fission type units don't go in comdat groups, since their linker is debug
; aware it's handled using the debug info semantics rather than raw ELF object
; semantics.
; OBJ_FISSION: Name: .debug_types.dwo (
; OBJ_FISSION-NOT: SHF_GROUP
; OBJ_FISSION: }

%struct.bar = type { i8 }
%"class.echidna::capybara::mongoose::fluffy" = type { i32, i32 }
%"struct.<anonymous namespace>::walrus" = type { i8 }
%struct.wombat = type { %struct.anon }
%struct.anon = type { i32, i32 }
%struct.baz = type { i8 }

@b = global %struct.bar zeroinitializer, align 1
@_ZN7echidna8capybara8mongoose6animalE = global %"class.echidna::capybara::mongoose::fluffy" zeroinitializer, align 4
@w = internal global %"struct.<anonymous namespace>::walrus" zeroinitializer, align 1
@wom = global %struct.wombat zeroinitializer, align 4
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

; Function Attrs: nounwind uwtable
define void @_Z3foov() #0 {
entry:
  %b = alloca %struct.baz, align 1
  call void @llvm.dbg.declare(metadata !{%struct.baz* %b}, metadata !46), !dbg !48
  ret void, !dbg !49
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

define internal void @__cxx_global_var_init() section ".text.startup" {
entry:
  call void @_ZN12_GLOBAL__N_16walrusC2Ev(%"struct.<anonymous namespace>::walrus"* @w), !dbg !50
  ret void, !dbg !50
}

; Function Attrs: nounwind uwtable
define internal void @_ZN12_GLOBAL__N_16walrusC2Ev(%"struct.<anonymous namespace>::walrus"* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %"struct.<anonymous namespace>::walrus"*, align 8
  store %"struct.<anonymous namespace>::walrus"* %this, %"struct.<anonymous namespace>::walrus"** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%"struct.<anonymous namespace>::walrus"** %this.addr}, metadata !51), !dbg !53
  %this1 = load %"struct.<anonymous namespace>::walrus"** %this.addr
  ret void, !dbg !54
}

define internal void @_GLOBAL__I_a() section ".text.startup" {
entry:
  call void @__cxx_global_var_init(), !dbg !55
  ret void, !dbg !55
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!43, !44}
!llvm.ident = !{!45}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5 ", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !21, metadata !38, metadata !2, metadata !"bar.dwo"} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/bar.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"bar.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !6, metadata !14, metadata !17}
!4 = metadata !{i32 786451, metadata !5, null, metadata !"bar", i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, null, metadata !"_ZTS3bar"} ; [ DW_TAG_structure_type ] [bar] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !"bar.h", metadata !"/tmp/dbginfo"}
!6 = metadata !{i32 786434, metadata !1, metadata !7, metadata !"fluffy", i32 13, i64 64, i64 32, i32 0, i32 0, null, metadata !10, i32 0, null, null, metadata !"_ZTSN7echidna8capybara8mongoose6fluffyE"} ; [ DW_TAG_class_type ] [fluffy] [line 13, size 64, align 32, offset 0] [def] [from ]
!7 = metadata !{i32 786489, metadata !1, metadata !8, metadata !"mongoose", i32 12} ; [ DW_TAG_namespace ] [mongoose] [line 12]
!8 = metadata !{i32 786489, metadata !1, metadata !9, metadata !"capybara", i32 11} ; [ DW_TAG_namespace ] [capybara] [line 11]
!9 = metadata !{i32 786489, metadata !1, null, metadata !"echidna", i32 10} ; [ DW_TAG_namespace ] [echidna] [line 10]
!10 = metadata !{metadata !11, metadata !13}
!11 = metadata !{i32 786445, metadata !1, metadata !"_ZTSN7echidna8capybara8mongoose6fluffyE", metadata !"a", i32 14, i64 32, i64 32, i64 0, i32 1, metadata !12} ; [ DW_TAG_member ] [a] [line 14, size 32, align 32, offset 0] [private] [from int]
!12 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!13 = metadata !{i32 786445, metadata !1, metadata !"_ZTSN7echidna8capybara8mongoose6fluffyE", metadata !"b", i32 15, i64 32, i64 32, i64 32, i32 1, metadata !12} ; [ DW_TAG_member ] [b] [line 15, size 32, align 32, offset 32] [private] [from int]
!14 = metadata !{i32 786451, metadata !1, null, metadata !"wombat", i32 31, i64 64, i64 32, i32 0, i32 0, null, metadata !15, i32 0, null, null, metadata !"_ZTS6wombat"} ; [ DW_TAG_structure_type ] [wombat] [line 31, size 64, align 32, offset 0] [def] [from ]
!15 = metadata !{metadata !16}
!16 = metadata !{i32 786445, metadata !1, metadata !"_ZTS6wombat", metadata !"a_b", i32 35, i64 64, i64 32, i64 0, i32 0, metadata !"_ZTSN6wombatUt_E"} ; [ DW_TAG_member ] [a_b] [line 35, size 64, align 32, offset 0] [from _ZTSN6wombatUt_E]
!17 = metadata !{i32 786451, metadata !1, metadata !"_ZTS6wombat", metadata !"", i32 32, i64 64, i64 32, i32 0, i32 0, null, metadata !18, i32 0, null, null, metadata !"_ZTSN6wombatUt_E"} ; [ DW_TAG_structure_type ] [line 32, size 64, align 32, offset 0] [def] [from ]
!18 = metadata !{metadata !19, metadata !20}
!19 = metadata !{i32 786445, metadata !1, metadata !"_ZTSN6wombatUt_E", metadata !"a", i32 33, i64 32, i64 32, i64 0, i32 0, metadata !12} ; [ DW_TAG_member ] [a] [line 33, size 32, align 32, offset 0] [from int]
!20 = metadata !{i32 786445, metadata !1, metadata !"_ZTSN6wombatUt_E", metadata !"b", i32 34, i64 32, i64 32, i64 32, i32 0, metadata !12} ; [ DW_TAG_member ] [b] [line 34, size 32, align 32, offset 32] [from int]
!21 = metadata !{metadata !22, metadata !26, metadata !27, metadata !36}
!22 = metadata !{i32 786478, metadata !1, metadata !23, metadata !"foo", metadata !"foo", metadata !"_Z3foov", i32 5, metadata !24, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z3foov, null, null, metadata !2, i32 5} ; [ DW_TAG_subprogram ] [line 5] [def] [foo]
!23 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/bar.cpp]
!24 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !25, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!25 = metadata !{null}
!26 = metadata !{i32 786478, metadata !1, metadata !23, metadata !"__cxx_global_var_init", metadata !"__cxx_global_var_init", metadata !"", i32 29, metadata !24, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @__cxx_global_var_init, null, null, metadata !2, i32 29} ; [ DW_TAG_subprogram ] [line 29] [local] [def] [__cxx_global_var_init]
!27 = metadata !{i32 786478, metadata !1, metadata !28, metadata !"walrus", metadata !"walrus", metadata !"_ZN12_GLOBAL__N_16walrusC2Ev", i32 25, metadata !32, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%"struct.<anonymous namespace>::walrus"*)* @_ZN12_GLOBAL__N_16walrusC2Ev, null, metadata !31, metadata !2, i32 25} ; [ DW_TAG_subprogram ] [line 25] [local] [def] [walrus]
!28 = metadata !{i32 786451, metadata !1, metadata !29, metadata !"walrus", i32 24, i64 8, i64 8, i32 0, i32 0, null, metadata !30, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [walrus] [line 24, size 8, align 8, offset 0] [def] [from ]
!29 = metadata !{i32 786489, metadata !1, null, metadata !"", i32 23} ; [ DW_TAG_namespace ] [line 23]
!30 = metadata !{metadata !31}
!31 = metadata !{i32 786478, metadata !1, metadata !28, metadata !"walrus", metadata !"walrus", metadata !"", i32 25, metadata !32, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !35, i32 25} ; [ DW_TAG_subprogram ] [line 25] [walrus]
!32 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !33, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!33 = metadata !{null, metadata !34}
!34 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !28} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from walrus]
!35 = metadata !{i32 786468}
!36 = metadata !{i32 786478, metadata !1, metadata !23, metadata !"", metadata !"", metadata !"_GLOBAL__I_a", i32 25, metadata !37, i1 true, i1 true, i32 0, i32 0, null, i32 64, i1 false, void ()* @_GLOBAL__I_a, null, null, metadata !2, i32 25} ; [ DW_TAG_subprogram ] [line 25] [local] [def]
!37 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!38 = metadata !{metadata !39, metadata !40, metadata !41, metadata !42}
!39 = metadata !{i32 786484, i32 0, null, metadata !"b", metadata !"b", metadata !"", metadata !23, i32 3, metadata !4, i32 0, i32 1, %struct.bar* @b, null} ; [ DW_TAG_variable ] [b] [line 3] [def]
!40 = metadata !{i32 786484, i32 0, metadata !7, metadata !"animal", metadata !"animal", metadata !"_ZN7echidna8capybara8mongoose6animalE", metadata !23, i32 18, metadata !6, i32 0, i32 1, %"class.echidna::capybara::mongoose::fluffy"* @_ZN7echidna8capybara8mongoose6animalE, null} ; [ DW_TAG_variable ] [animal] [line 18] [def]
!41 = metadata !{i32 786484, i32 0, null, metadata !"w", metadata !"w", metadata !"", metadata !23, i32 29, metadata !28, i32 1, i32 1, %"struct.<anonymous namespace>::walrus"* @w, null} ; [ DW_TAG_variable ] [w] [line 29] [local] [def]
!42 = metadata !{i32 786484, i32 0, null, metadata !"wom", metadata !"wom", metadata !"", metadata !23, i32 38, metadata !14, i32 0, i32 1, %struct.wombat* @wom, null} ; [ DW_TAG_variable ] [wom] [line 38] [def]
!43 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!44 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!45 = metadata !{metadata !"clang version 3.5 "}
!46 = metadata !{i32 786688, metadata !22, metadata !"b", metadata !23, i32 7, metadata !47, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [b] [line 7]
!47 = metadata !{i32 786451, metadata !1, metadata !22, metadata !"baz", i32 6, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [baz] [line 6, size 8, align 8, offset 0] [def] [from ]
!48 = metadata !{i32 7, i32 0, metadata !22, null}
!49 = metadata !{i32 8, i32 0, metadata !22, null} ; [ DW_TAG_imported_declaration ]
!50 = metadata !{i32 29, i32 0, metadata !26, null}
!51 = metadata !{i32 786689, metadata !27, metadata !"this", null, i32 16777216, metadata !52, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!52 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !28} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from walrus]
!53 = metadata !{i32 0, i32 0, metadata !27, null}
!54 = metadata !{i32 25, i32 0, metadata !27, null}
!55 = metadata !{i32 25, i32 0, metadata !36, null}
