; RUN: llc -split-dwarf=Enable -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck %s
; RUN: llvm-readobj --relocations %t | FileCheck --check-prefix=CHECK-RELOCS %s

; From:
; class A {
; public:
;   A(int i = 0) : a(i) {}
; private:
;   int a;
; };
;
; A a;

; With function sections enabled make sure that we have a DW_AT_ranges attribute.
; CHECK: DW_AT_ranges

; Check that we have a relocation against the .debug_ranges section.
; CHECK-RELOCS: R_X86_64_32 .debug_ranges 0x0

%class.A = type { i32 }

@a = global %class.A zeroinitializer, align 4
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

define internal void @__cxx_global_var_init() section ".text.startup" {
entry:
  call void @_ZN1AC2Ei(%class.A* @a, i32 0), !dbg !26
  ret void, !dbg !26
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZN1AC2Ei(%class.A* %this, i32 %i) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  %i.addr = alloca i32, align 4
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !27), !dbg !29
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %i.addr}, metadata !30), !dbg !31
  %this1 = load %class.A** %this.addr
  %a = getelementptr inbounds %class.A* %this1, i32 0, i32 0, !dbg !31
  %0 = load i32* %i.addr, align 4, !dbg !31
  store i32 %0, i32* %a, align 4, !dbg !31
  ret void, !dbg !31
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

define internal void @_GLOBAL__I_a() section ".text.startup" {
entry:
  call void @__cxx_global_var_init(), !dbg !32
  ret void, !dbg !32
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23, !24}
!llvm.ident = !{!25}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5 (trunk 199923) (llvm/trunk 199940)", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !13, metadata !21, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/baz.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"baz.cpp", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786434, metadata !1, null, metadata !"A", i32 1, i64 32, i64 32, i32 0, i32 0, null, metadata !5, i32 0, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 1, size 32, align 32, offset 0] [def] [from ]
!5 = metadata !{metadata !6, metadata !8}
!6 = metadata !{i32 786445, metadata !1, metadata !"_ZTS1A", metadata !"a", i32 5, i64 32, i64 32, i64 0, i32 1, metadata !7} ; [ DW_TAG_member ] [a] [line 5, size 32, align 32, offset 0] [private] [from int]
!7 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"A", metadata !"A", metadata !"", i32 3, metadata !9, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !12, i32 3} ; [ DW_TAG_subprogram ] [line 3] [A]
!9 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{null, metadata !11, metadata !7}
!11 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!12 = metadata !{i32 786468}
!13 = metadata !{metadata !14, metadata !18, metadata !19}
!14 = metadata !{i32 786478, metadata !1, metadata !15, metadata !"__cxx_global_var_init", metadata !"__cxx_global_var_init", metadata !"", i32 8, metadata !16, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @__cxx_global_var_init, null, null, metadata !2, i32 8} ; [ DW_TAG_subprogram ] [line 8] [local] [def] [__cxx_global_var_init]
!15 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/baz.cpp]
!16 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !17, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = metadata !{null}
!18 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"A", metadata !"A", metadata !"_ZN1AC2Ei", i32 3, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.A*, i32)* @_ZN1AC2Ei, null, metadata !8, metadata !2, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [A]
!19 = metadata !{i32 786478, metadata !1, metadata !15, metadata !"", metadata !"", metadata !"_GLOBAL__I_a", i32 3, metadata !20, i1 true, i1 true, i32 0, i32 0, null, i32 64, i1 false, void ()* @_GLOBAL__I_a, null, null, metadata !2, i32 3} ; [ DW_TAG_subprogram ] [line 3] [local] [def]
!20 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = metadata !{metadata !22}
!22 = metadata !{i32 786484, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !15, i32 8, metadata !4, i32 0, i32 1, %class.A* @a, null} ; [ DW_TAG_variable ] [a] [line 8] [def]
!23 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!24 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!25 = metadata !{metadata !"clang version 3.5 (trunk 199923) (llvm/trunk 199940)"}
!26 = metadata !{i32 8, i32 0, metadata !14, null} ; [ DW_TAG_imported_declaration ]
!27 = metadata !{i32 786689, metadata !18, metadata !"this", null, i32 16777216, metadata !28, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!28 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1A]
!29 = metadata !{i32 0, i32 0, metadata !18, null}
!30 = metadata !{i32 786689, metadata !18, metadata !"i", metadata !15, i32 33554435, metadata !7, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [i] [line 3]
!31 = metadata !{i32 3, i32 0, metadata !18, null}
!32 = metadata !{i32 3, i32 0, metadata !19, null}
