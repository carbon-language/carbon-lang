; REQUIRES: object-emission

; RUN: llc %s -o %t -filetype=obj -O0 -generate-odr-hash -mtriple=x86_64-unknown-linux-gnu
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
;
; Generated from:
; struct bar {};

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

; Check that we generate a hash for bar and the value.
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: debug_str{{.*}}"bar"
; CHECK: DW_AT_GNU_odr_signature [DW_FORM_data8] (0x200520c0d5b90eff)
; CHECK: DW_TAG_namespace
; CHECK-NEXT: debug_str{{.*}}"echidna"
; CHECK: DW_TAG_namespace
; CHECK-NEXT: debug_str{{.*}}"capybara"
; CHECK: DW_TAG_namespace
; CHECK-NEXT: debug_str{{.*}}"mongoose"
; CHECK: DW_TAG_class_type
; CHECK-NEXT: debug_str{{.*}}"fluffy"
; CHECK: DW_AT_GNU_odr_signature [DW_FORM_data8]   (0x9a0124d5a0c21c52)

; We emit no hash for walrus since the type is contained in an anonymous
; namespace and won't violate any ODR-ness.
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: debug_str{{.*}}"walrus"
; CHECK-NEXT: DW_AT_byte_size
; CHECK-NEXT: DW_AT_decl_file
; CHECK-NEXT: DW_AT_decl_line
; CHECK-NOT: DW_AT_GNU_odr_signature
; CHECK: DW_TAG_subprogram

; Check that we generate a hash for wombat and the value, but not for the
; anonymous type contained within.
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: debug_str{{.*}}wombat
; CHECK: DW_AT_GNU_odr_signature [DW_FORM_data8] (0x685bcc220141e9d7)
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_byte_size
; CHECK-NEXT: DW_AT_decl_file
; CHECK-NEXT: DW_AT_decl_line
; CHECK: DW_TAG_member
; CHECK-NEXT: debug_str{{.*}}"a"

; Check that we don't generate a hash for baz.
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: debug_str{{.*}}"baz"
; CHECK-NOT: DW_AT_GNU_odr_signature

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

@_ZN12_GLOBAL__N_16walrusC1Ev = alias internal void (%"struct.<anonymous namespace>::walrus"*)* @_ZN12_GLOBAL__N_16walrusC2Ev

; Function Attrs: nounwind uwtable
define void @_Z3foov() #0 {
entry:
  %b = alloca %struct.baz, align 1
  call void @llvm.dbg.declare(metadata !{%struct.baz* %b}, metadata !63), !dbg !71
  ret void, !dbg !72
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

define internal void @__cxx_global_var_init() section ".text.startup" {
entry:
  call void @_ZN12_GLOBAL__N_16walrusC1Ev(%"struct.<anonymous namespace>::walrus"* @w), !dbg !73
  ret void, !dbg !73
}

; Function Attrs: nounwind uwtable
define internal void @_ZN12_GLOBAL__N_16walrusC2Ev(%"struct.<anonymous namespace>::walrus"* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %"struct.<anonymous namespace>::walrus"*, align 8
  store %"struct.<anonymous namespace>::walrus"* %this, %"struct.<anonymous namespace>::walrus"** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%"struct.<anonymous namespace>::walrus"** %this.addr}, metadata !74), !dbg !76
  %this1 = load %"struct.<anonymous namespace>::walrus"** %this.addr
  ret void, !dbg !76
}

define internal void @_GLOBAL__I_a() section ".text.startup" {
entry:
  call void @__cxx_global_var_init(), !dbg !77
  ret void, !dbg !77
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!62}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 (trunk 187387) (llvm/trunk 187385)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !20, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/bar.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"bar.cpp", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !8, metadata !9, metadata !18}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"foo", metadata !"foo", metadata !"_Z3foov", i32 6, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z3foov, null, null, metadata !2, i32 6} ; [ DW_TAG_subprogram ] [line 6] [def] [foo]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/bar.cpp]
!6 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"__cxx_global_var_init", metadata !"__cxx_global_var_init", metadata !"", i32 31, metadata !6, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @__cxx_global_var_init, null, null, metadata !2, i32 31} ; [ DW_TAG_subprogram ] [line 31] [local] [def] [__cxx_global_var_init]
!9 = metadata !{i32 786478, metadata !1, metadata !10, metadata !"walrus", metadata !"walrus", metadata !"_ZN12_GLOBAL__N_16walrusC2Ev", i32 27, metadata !11, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%"struct.<anonymous namespace>::walrus"*)* @_ZN12_GLOBAL__N_16walrusC2Ev, null, metadata !16, metadata !2, i32 27} ; [ DW_TAG_subprogram ] [line 27] [local] [def] [walrus]
!10 = metadata !{i32 786489, metadata !1, null, metadata !"", i32 25} ; [ DW_TAG_namespace ] [line 25]
!11 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{null, metadata !13}
!13 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !14} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from walrus]
!14 = metadata !{i32 786451, metadata !1, metadata !10, metadata !"walrus", i32 26, i64 8, i64 8, i32 0, i32 0, null, metadata !15, i32 0, null, null} ; [ DW_TAG_structure_type ] [walrus] [line 26, size 8, align 8, offset 0] [def] [from ]
!15 = metadata !{metadata !16}
!16 = metadata !{i32 786478, metadata !1, metadata !14, metadata !"walrus", metadata !"walrus", metadata !"", i32 27, metadata !11, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !17, i32 27} ; [ DW_TAG_subprogram ] [line 27] [walrus]
!17 = metadata !{i32 786468}
!18 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"_GLOBAL__I_a", metadata !"_GLOBAL__I_a", metadata !"", i32 27, metadata !19, i1 true, i1 true, i32 0, i32 0, null, i32 64, i1 false, void ()* @_GLOBAL__I_a, null, null, metadata !2, i32 27} ; [ DW_TAG_subprogram ] [line 27] [local] [def] [_GLOBAL__I_a]
!19 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!20 = metadata !{metadata !21, metadata !29, metadata !43, metadata !44}
!21 = metadata !{i32 786484, i32 0, null, metadata !"b", metadata !"b", metadata !"", metadata !5, i32 4, metadata !22, i32 0, i32 1, %struct.bar* @b, null} ; [ DW_TAG_variable ] [b] [line 4] [def]
!22 = metadata !{i32 786451, metadata !1, null, metadata !"bar", i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !23, i32 0, null, null} ; [ DW_TAG_structure_type ] [bar] [line 1, size 8, align 8, offset 0] [def] [from ]
!23 = metadata !{metadata !24}
!24 = metadata !{i32 786478, metadata !1, metadata !22, metadata !"bar", metadata !"bar", metadata !"", i32 1, metadata !25, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !28, i32 1} ; [ DW_TAG_subprogram ] [line 1] [bar]
!25 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !26, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!26 = metadata !{null, metadata !27}
!27 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !22} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from bar]
!28 = metadata !{i32 786468}
!29 = metadata !{i32 786484, i32 0, metadata !30, metadata !"animal", metadata !"animal", metadata !"_ZN7echidna8capybara8mongoose6animalE", metadata !5, i32 20, metadata !33, i32 0, i32 1, %"class.echidna::capybara::mongoose::fluffy"* @_ZN7echidna8capybara8mongoose6animalE, null} ; [ DW_TAG_variable ] [animal] [line 20] [def]
!30 = metadata !{i32 786489, metadata !1, metadata !31, metadata !"mongoose", i32 14} ; [ DW_TAG_namespace ] [mongoose] [line 14]
!31 = metadata !{i32 786489, metadata !1, metadata !32, metadata !"capybara", i32 13} ; [ DW_TAG_namespace ] [capybara] [line 13]
!32 = metadata !{i32 786489, metadata !1, null, metadata !"echidna", i32 12} ; [ DW_TAG_namespace ] [echidna] [line 12]
!33 = metadata !{i32 786434, metadata !1, metadata !30, metadata !"fluffy", i32 15, i64 64, i64 32, i32 0, i32 0, null, metadata !34, i32 0, null, null} ; [ DW_TAG_class_type ] [fluffy] [line 15, size 64, align 32, offset 0] [def] [from ]
!34 = metadata !{metadata !35, metadata !37, metadata !38}
!35 = metadata !{i32 786445, metadata !1, metadata !33, metadata !"a", i32 16, i64 32, i64 32, i64 0, i32 1, metadata !36} ; [ DW_TAG_member ] [a] [line 16, size 32, align 32, offset 0] [private] [from int]
!36 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!37 = metadata !{i32 786445, metadata !1, metadata !33, metadata !"b", i32 17, i64 32, i64 32, i64 32, i32 1, metadata !36} ; [ DW_TAG_member ] [b] [line 17, size 32, align 32, offset 32] [private] [from int]
!38 = metadata !{i32 786478, metadata !1, metadata !33, metadata !"fluffy", metadata !"fluffy", metadata !"", i32 15, metadata !39, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !42, i32 15} ; [ DW_TAG_subprogram ] [line 15] [fluffy]
!39 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !40, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!40 = metadata !{null, metadata !41}
!41 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !33} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from fluffy]
!42 = metadata !{i32 786468}
!43 = metadata !{i32 786484, i32 0, null, metadata !"w", metadata !"w", metadata !"", metadata !5, i32 31, metadata !14, i32 1, i32 1, %"struct.<anonymous namespace>::walrus"* @w, null} ; [ DW_TAG_variable ] [w] [line 31] [local] [def]
!44 = metadata !{i32 786484, i32 0, null, metadata !"wom", metadata !"wom", metadata !"", metadata !5, i32 40, metadata !45, i32 0, i32 1, %struct.wombat* @wom, null} ; [ DW_TAG_variable ] [wom] [line 40] [def]
!45 = metadata !{i32 786451, metadata !1, null, metadata !"wombat", i32 33, i64 64, i64 32, i32 0, i32 0, null, metadata !46, i32 0, null, null} ; [ DW_TAG_structure_type ] [wombat] [line 33, size 64, align 32, offset 0] [def] [from ]
!46 = metadata !{metadata !47, metadata !57}
!47 = metadata !{i32 786445, metadata !1, metadata !45, metadata !"a_b", i32 37, i64 64, i64 32, i64 0, i32 0, metadata !48} ; [ DW_TAG_member ] [a_b] [line 37, size 64, align 32, offset 0] [from ]
!48 = metadata !{i32 786451, metadata !1, metadata !45, metadata !"", i32 34, i64 64, i64 32, i32 0, i32 0, null, metadata !49, i32 0, null, null} ; [ DW_TAG_structure_type ] [line 34, size 64, align 32, offset 0] [def] [from ]
!49 = metadata !{metadata !50, metadata !51, metadata !52}
!50 = metadata !{i32 786445, metadata !1, metadata !48, metadata !"a", i32 35, i64 32, i64 32, i64 0, i32 0, metadata !36} ; [ DW_TAG_member ] [a] [line 35, size 32, align 32, offset 0] [from int]
!51 = metadata !{i32 786445, metadata !1, metadata !48, metadata !"b", i32 36, i64 32, i64 32, i64 32, i32 0, metadata !36} ; [ DW_TAG_member ] [b] [line 36, size 32, align 32, offset 32] [from int]
!52 = metadata !{i32 786478, metadata !1, metadata !48, metadata !"", metadata !"", metadata !"", i32 34, metadata !53, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !56, i32 34} ; [ DW_TAG_subprogram ] [line 34]
!53 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !54, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!54 = metadata !{null, metadata !55}
!55 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !48} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from ]
!56 = metadata !{i32 786468}
!57 = metadata !{i32 786478, metadata !1, metadata !45, metadata !"wombat", metadata !"wombat", metadata !"", i32 33, metadata !58, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !61, i32 33} ; [ DW_TAG_subprogram ] [line 33] [wombat]
!58 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !59, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!59 = metadata !{null, metadata !60}
!60 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !45} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from wombat]
!61 = metadata !{i32 786468}
!62 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
!63 = metadata !{i32 786688, metadata !4, metadata !"b", metadata !5, i32 9, metadata !64, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [b] [line 9]
!64 = metadata !{i32 786451, metadata !1, metadata !4, metadata !"baz", i32 7, i64 8, i64 8, i32 0, i32 0, null, metadata !65, i32 0, null, null} ; [ DW_TAG_structure_type ] [baz] [line 7, size 8, align 8, offset 0] [def] [from ]
!65 = metadata !{metadata !66}
!66 = metadata !{i32 786478, metadata !1, metadata !64, metadata !"baz", metadata !"baz", metadata !"", i32 7, metadata !67, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !70, i32 7} ; [ DW_TAG_subprogram ] [line 7] [baz]
!67 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !68, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!68 = metadata !{null, metadata !69}
!69 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !64} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from baz]
!70 = metadata !{i32 786468}
!71 = metadata !{i32 9, i32 0, metadata !4, null}
!72 = metadata !{i32 10, i32 0, metadata !4, null}
!73 = metadata !{i32 31, i32 0, metadata !8, null}
!74 = metadata !{i32 786689, metadata !9, metadata !"this", metadata !5, i32 16777243, metadata !75, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 27]
!75 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !14} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from walrus]
!76 = metadata !{i32 27, i32 0, metadata !9, null}
!77 = metadata !{i32 27, i32 0, metadata !18, null}
