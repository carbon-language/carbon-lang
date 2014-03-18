; REQUIRES: object-emission
;
; RUN: llvm-link %s %p/type-unique-odr-b.ll -S -o - | %llc_dwarf -filetype=obj -O0 | llvm-dwarfdump -debug-dump=info - | FileCheck %s
;
; Test ODR-based type uniquing for C++ class members.
; rdar://problem/15851313.
;
; $ cat -n type-unique-odr-a.cpp
;     1	class A {
;     2	  int data;
;     3	protected:
;     4	  void getFoo();
;     5	};
;     6
;     7	static void bar() {
;     8	  A a;
;     9	}
;    10
;    11	void baz() { bar(); }
;; #include "ab.h"
; foo_t bar() {
;     return A().getFoo();
; }
;
; CHECK:      DW_TAG_subprogram
; CHECK-NEXT:   DW_AT_MIPS_linkage_name {{.*}} "_Z3bazv"
; CHECK:      DW_TAG_subprogram
; CHECK-NEXT:   DW_AT_MIPS_linkage_name {{.*}} "_ZL3barv"
; CHECK:      DW_TAG_class_type
; CHECK-NEXT:   DW_AT_name {{.*}} "A"
; CHECK-NOT:  DW_TAG
; CHECK:      DW_TAG_member
; CHECK-NEXT:   DW_AT_name {{.*}} "data"
; CHECK-NOT:  DW_TAG
; CHECK:      DW_TAG_subprogram
; CHECK-NEXT:   DW_AT_MIPS_linkage_name {{.*}} "_ZN1A6getFooEv"
; CHECK-NEXT:   DW_AT_name {{.*}} "getFoo"

; getFoo and A may only appear once.
; CHECK-NOT:  {{(getFoo)|("A")}}


; ModuleID = 'type-unique-odr-a.cpp'

%class.A = type { i32 }

; Function Attrs: nounwind
define void @_Z3bazv() #0 {
entry:
  call void @_ZL3barv(), !dbg !23
  ret void, !dbg !23
}

; Function Attrs: nounwind
define internal void @_ZL3barv() #0 {
entry:
  %a = alloca %class.A, align 4
  call void @llvm.dbg.declare(metadata !{%class.A* %a}, metadata !24), !dbg !25
  ret void, !dbg !26
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20, !21}
!llvm.ident = !{!22}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !14, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [<unknown>] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"<unknown>", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786434, metadata !5, null, metadata !"A", i32 1, i64 32, i64 32, i32 0, i32 0, null, metadata !6, i32 0, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 1, size 32, align 32, offset 0] [def] [from ]
!5 = metadata !{metadata !"type-unique-odr-a.cpp", metadata !""}
!6 = metadata !{metadata !7, metadata !9}
!7 = metadata !{i32 786445, metadata !5, metadata !"_ZTS1A", metadata !"data", i32 2, i64 32, i64 32, i64 0, i32 1, metadata !8} ; [ DW_TAG_member ] [data] [line 2, size 32, align 32, offset 0] [private] [from int]
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786478, metadata !5, metadata !"_ZTS1A", metadata !"getFoo", metadata !"getFoo", metadata !"_ZN1A6getFooEv", i32 4, metadata !10, i1 false, i1 false, i32 0, i32 0, null, i32 258, i1 false, null, null, i32 0, metadata !13, i32 4} ; [ DW_TAG_subprogram ] [line 4] [protected] [getFoo]
!10 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = metadata !{null, metadata !12}
!12 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!13 = metadata !{i32 786468}
!14 = metadata !{metadata !15, metadata !19}
!15 = metadata !{i32 786478, metadata !5, metadata !16, metadata !"baz", metadata !"baz", metadata !"_Z3bazv", i32 11, metadata !17, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z3bazv, null, null, metadata !2, i32 11} ; [ DW_TAG_subprogram ] [line 11] [def] [baz]
!16 = metadata !{i32 786473, metadata !5}         ; [ DW_TAG_file_type ] [type-unique-odr-a.cpp]
!17 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !18, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{null}
!19 = metadata !{i32 786478, metadata !5, metadata !16, metadata !"bar", metadata !"bar", metadata !"_ZL3barv", i32 7, metadata !17, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_ZL3barv, null, null, metadata !2, i32 7} ; [ DW_TAG_subprogram ] [line 7] [local] [def] [bar]
!20 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!22 = metadata !{metadata !"clang version 3.5.0 "}
!23 = metadata !{i32 11, i32 0, metadata !15, null}
!24 = metadata !{i32 786688, metadata !19, metadata !"a", metadata !16, i32 8, metadata !"_ZTS1A", i32 0, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 8]
!25 = metadata !{i32 8, i32 0, metadata !19, null} ; [ DW_TAG_imported_declaration ]
!26 = metadata !{i32 9, i32 0, metadata !19, null}
