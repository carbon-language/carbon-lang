; RUN: llc -split-dwarf=Enable -O0 < %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Test the emission of gmlt-like inlining information into the skeleton unit.
; This allows inline-aware symbolication/backtracing given only the linked
; executable, without needing access to the .dwos.

; A simple example of inlining generated with clang -gsplit-dwarf

; A member function is used to force emission of the declaration of the
; function into the .dwo file, which may be shared with other CUs in the dwo ;
; under fission, but should not be shared with the skeleton's CU. This also
; tests the general case of context emission, which is suppressed in gmlt-like
; data.

; Include a template just to test template parameters are not emitted in
; gmlt-like data.

; And some varargs to make sure DW_TAG_unspecified_parameters is not emitted.

; And a using declaration in a nested lexical_block... because that shouldn't
; be emitted either.

; Minor complication: after generating the LLVM IR, it was manually edited so
; that the 'f1()' call from f3 was reordered to appear between the two inlined
; f1 calls from f2. This causes f2's inlined_subroutine to use DW_AT_ranges,
; thus exercising range list generation/referencing which was buggy.

; struct foo {
;   template<typename T>
;   static void f2();
;   static void f3(...);
; };
;
; void f1();
;
; template<typename T>
; inline __attribute__((always_inline)) void foo::f2() {
;   f1();
;   f1();
; }
;
; void foo::f3(...) {
;   if (true) {
;     f1();
;     f2<int>();
;     using ::foo;
;   }
; }

; Check that we emit the usual gmlt-like data for this file, including brief
; descriptions of subprograms with inlined scopes.

; FIXME: Once tools support indexed addresses in the skeleton CU, we should use
; those (DW_FORM_addr would become DW_FORM_GNU_addr_index below) since those
; addresses will already be in the address pool anyway.

; CHECK:      DW_TAG_subprogram
; CHECK-NEXT:   DW_AT_name {{.*}} "f2<int>"
; CHECK-NOT: DW_
; CHECK:      DW_TAG_subprogram
; CHECK-NEXT:   DW_AT_low_pc [DW_FORM_addr]
; CHECK-NEXT:   DW_AT_high_pc
; CHECK-NEXT:   DW_AT_name {{.*}} "f3"
; CHECK-NOT: {{DW_|NULL}}
; CHECK:        DW_TAG_inlined_subroutine
; CHECK-NEXT:     DW_AT_abstract_origin {{.*}} "f2<int>"
; CHECK-NEXT:     DW_AT_ranges
; CHECK-NEXT:     DW_AT_call_file
; CHECK-NEXT:     DW_AT_call_line {{.*}} (18)
; CHECK-NOT: DW_

; Function Attrs: uwtable
define void @_ZN3foo2f3Ez(...) #0 align 2 {
entry:
  call void @_Z2f1v(), !dbg !26
  call void @_Z2f1v(), !dbg !25
  call void @_Z2f1v(), !dbg !28
  ret void, !dbg !29
}

declare void @_Z2f1v() #1

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}
!llvm.ident = !{!24}

!0 = metadata !{metadata !"0x11\004\00clang version 3.6.0 \000\00\000\00fission-inline.dwo\001", metadata !1, metadata !2, metadata !3, metadata !9, metadata !2, metadata !18} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/fission-inline.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"fission-inline.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x13\00foo\001\008\008\000\000\000", metadata !1, null, null, metadata !5, null, null, metadata !"_ZTS3foo"} ; [ DW_TAG_structure_type ] [foo] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{metadata !"0x2e\00f3\00f3\00_ZN3foo2f3Ez\004\000\000\000\000\00256\000\004", metadata !1, metadata !"_ZTS3foo", metadata !7, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 4] [f3]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, null}
!9 = metadata !{metadata !10, metadata !11}
!10 = metadata !{metadata !"0x2e\00f3\00f3\00_ZN3foo2f3Ez\0015\000\001\000\000\00256\000\0015", metadata !1, metadata !"_ZTS3foo", metadata !7, null, void (...)* @_ZN3foo2f3Ez, null, metadata !6, metadata !2} ; [ DW_TAG_subprogram ] [line 15] [def] [f3]
!11 = metadata !{metadata !"0x2e\00f2<int>\00f2<int>\00_ZN3foo2f2IiEEvv\0010\000\001\000\000\00256\000\0010", metadata !1, metadata !"_ZTS3foo", metadata !12, null, null, metadata !14, metadata !17, metadata !2} ; [ DW_TAG_subprogram ] [line 10] [def] [f2<int>]
!12 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !13, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = metadata !{null}
!14 = metadata !{metadata !15}
!15 = metadata !{metadata !"0x2f\00T\000\000", null, metadata !16, null} ; [ DW_TAG_template_type_parameter ]
!16 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!17 = metadata !{metadata !"0x2e\00f2<int>\00f2<int>\00_ZN3foo2f2IiEEvv\0010\000\000\000\000\00256\000\0010", metadata !1, metadata !"_ZTS3foo", metadata !12, null, null, metadata !14, null, null} ; [ DW_TAG_subprogram ] [line 10] [f2<int>]
!18 = metadata !{metadata !19}
!19 = metadata !{metadata !"0x8\0019\00", metadata !20, metadata !"_ZTS3foo"} ; [ DW_TAG_imported_declaration ]
!20 = metadata !{metadata !"0xb\0016\0013\001", metadata !1, metadata !21} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/fission-inline.cpp]
!21 = metadata !{metadata !"0xb\0016\007\000", metadata !1, metadata !10} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/fission-inline.cpp]
!22 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!23 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!24 = metadata !{metadata !"clang version 3.6.0 "}
!25 = metadata !{i32 17, i32 5, metadata !20, null}
!26 = metadata !{i32 11, i32 3, metadata !11, metadata !27}
!27 = metadata !{i32 18, i32 5, metadata !20, null}
!28 = metadata !{i32 12, i32 3, metadata !11, metadata !27}
!29 = metadata !{i32 21, i32 1, metadata !10, null}
