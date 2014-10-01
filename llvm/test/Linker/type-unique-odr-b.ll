; RUN: true
; This file belongs to type-unique-odr-a.ll.
;
; Test ODR-based type uniquing for C++ class members.
; rdar://problem/15851313.
;
; $ cat -n type-unique-odr-b.cpp
;     1	// Make this declaration start on a different line.
;     2	class A {
;     3	  int data;
;     4	protected:
;     5	  void getFoo();
;     6	};
;     7
;     8	void A::getFoo() {}
;     9
;    10	static void bar() {}
;    11	void f() { bar(); };

; ModuleID = 'type-unique-odr-b.cpp'

%class.A = type { i32 }

; Function Attrs: nounwind
define void @_ZN1A6getFooEv(%class.A* %this) #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !24), !dbg !26
  %this1 = load %class.A** %this.addr
  ret void, !dbg !27
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind
define void @_Z1fv() #0 {
entry:
  call void @_ZL3barv(), !dbg !28
  ret void, !dbg !28
}

; Function Attrs: nounwind
define internal void @_ZL3barv() #0 {
entry:
  ret void, !dbg !29
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !14, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [<unknown>] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"<unknown>", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786434, metadata !5, null, metadata !"A", i32 2, i64 32, i64 32, i32 0, i32 0, null, metadata !6, i32 0, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 2, size 32, align 32, offset 0] [def] [from ]
!5 = metadata !{metadata !"type-unique-odr-b.cpp", metadata !""}
!6 = metadata !{metadata !7, metadata !9}
!7 = metadata !{i32 786445, metadata !5, metadata !"_ZTS1A", metadata !"data", i32 3, i64 32, i64 32, i64 0, i32 1, metadata !8} ; [ DW_TAG_member ] [data] [line 3, size 32, align 32, offset 0] [private] [from int]
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786478, metadata !5, metadata !"_ZTS1A", metadata !"getFoo", metadata !"getFoo", metadata !"_ZN1A6getFooEv", i32 5, metadata !10, i1 false, i1 false, i32 0, i32 0, null, i32 258, i1 false, null, null, i32 0, metadata !13, i32 5} ; [ DW_TAG_subprogram ] [line 5] [protected] [getFoo]
!10 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = metadata !{null, metadata !12}
!12 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!13 = metadata !{i32 786468}
!14 = metadata !{metadata !15, metadata !16, metadata !20}
!15 = metadata !{i32 786478, metadata !5, metadata !"_ZTS1A", metadata !"getFoo", metadata !"getFoo", metadata !"_ZN1A6getFooEv", i32 8, metadata !10, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.A*)* @_ZN1A6getFooEv, null, metadata !9, metadata !2, i32 8} ; [ DW_TAG_subprogram ] [line 8] [def] [getFoo]
!16 = metadata !{i32 786478, metadata !5, metadata !17, metadata !"f", metadata !"f", metadata !"_Z1fv", i32 11, metadata !18, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z1fv, null, null, metadata !2, i32 11} ; [ DW_TAG_subprogram ] [line 11] [def] [f]
!17 = metadata !{i32 786473, metadata !5}         ; [ DW_TAG_file_type ] [type-unique-odr-b.cpp]
!18 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !19, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!19 = metadata !{null}
!20 = metadata !{i32 786478, metadata !5, metadata !17, metadata !"bar", metadata !"bar", metadata !"_ZL3barv", i32 10, metadata !18, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_ZL3barv, null, null, metadata !2, i32 10} ; [ DW_TAG_subprogram ] [line 10] [local] [def] [bar]
!21 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!22 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!23 = metadata !{metadata !"clang version 3.5.0 "}
!24 = metadata !{i32 786689, metadata !15, metadata !"this", null, i32 16777216, metadata !25, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!25 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1A]
!26 = metadata !{i32 0, i32 0, metadata !15, null}
!27 = metadata !{i32 8, i32 0, metadata !15, null} ; [ DW_TAG_imported_declaration ]
!28 = metadata !{i32 11, i32 0, metadata !16, null}
!29 = metadata !{i32 10, i32 0, metadata !20, null}
