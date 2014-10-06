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
; CHECK:      DW_TAG_class_type
; CHECK-NEXT:   DW_AT_name {{.*}} "A"
; CHECK-NOT:  DW_TAG
; CHECK:      DW_TAG_member
; CHECK-NEXT:   DW_AT_name {{.*}} "data"
; CHECK-NOT:  DW_TAG
; CHECK:      DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_MIPS_linkage_name {{.*}} "_ZN1A6getFooEv"
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "getFoo"
; CHECK:      DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_MIPS_linkage_name {{.*}} "_Z3bazv"
; CHECK:      DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_MIPS_linkage_name {{.*}} "_ZL3barv"

; getFoo and A may only appear once.
; CHECK-NOT:  AT_name{{.*(getFoo)|("A")}}


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
  call void @llvm.dbg.declare(metadata !{%class.A* %a}, metadata !24, metadata !{metadata !"0x102"}), !dbg !25
  ret void, !dbg !26
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20, !21}
!llvm.ident = !{!22}

!0 = metadata !{metadata !"0x11\004\00clang version 3.5.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !3, metadata !14, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [<unknown>] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"<unknown>", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2\00A\001\0032\0032\000\000\000", metadata !5, null, null, metadata !6, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 1, size 32, align 32, offset 0] [def] [from ]
!5 = metadata !{metadata !"type-unique-odr-a.cpp", metadata !""}
!6 = metadata !{metadata !7, metadata !9}
!7 = metadata !{metadata !"0xd\00data\002\0032\0032\000\001", metadata !5, metadata !"_ZTS1A", metadata !8} ; [ DW_TAG_member ] [data] [line 2, size 32, align 32, offset 0] [private] [from int]
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !"0x2e\00getFoo\00getFoo\00_ZN1A6getFooEv\004\000\000\000\006\00258\000\004", metadata !5, metadata !"_ZTS1A", metadata !10, null, null, null, i32 0, metadata !13} ; [ DW_TAG_subprogram ] [line 4] [protected] [getFoo]
!10 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !11, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = metadata !{null, metadata !12}
!12 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", null, null, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!13 = metadata !{i32 786468}
!14 = metadata !{metadata !15, metadata !19}
!15 = metadata !{metadata !"0x2e\00baz\00baz\00_Z3bazv\0011\000\001\000\006\00256\000\0011", metadata !5, metadata !16, metadata !17, null, void ()* @_Z3bazv, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 11] [def] [baz]
!16 = metadata !{metadata !"0x29", metadata !5}         ; [ DW_TAG_file_type ] [type-unique-odr-a.cpp]
!17 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{null}
!19 = metadata !{metadata !"0x2e\00bar\00bar\00_ZL3barv\007\001\001\000\006\00256\000\007", metadata !5, metadata !16, metadata !17, null, void ()* @_ZL3barv, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 7] [local] [def] [bar]
!20 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!22 = metadata !{metadata !"clang version 3.5.0 "}
!23 = metadata !{i32 11, i32 0, metadata !15, null}
!24 = metadata !{metadata !"0x100\00a\008\000", metadata !19, metadata !16, metadata !"_ZTS1A"} ; [ DW_TAG_auto_variable ] [a] [line 8]
!25 = metadata !{i32 8, i32 0, metadata !19, null}
!26 = metadata !{i32 9, i32 0, metadata !19, null}
