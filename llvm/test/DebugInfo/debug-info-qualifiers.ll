; REQUIRES: object-emission
; Test (r)value qualifiers on C++11 non-static member functions.
; Generated from tools/clang/test/CodeGenCXX/debug-info-qualifiers.cpp
;
; class A {
; public:
;   void l() const &;
;   void r() const &&;
; };
;
; void g() {
;   A a;
;   auto pl = &A::l;
;   auto pr = &A::r;
; }
;
; RUN: %llc_dwarf -filetype=obj -O0 < %s | llvm-dwarfdump - | FileCheck %s
; CHECK: DW_TAG_subroutine_type     DW_CHILDREN_yes
; CHECK-NEXT: DW_AT_reference  DW_FORM_flag_present
; CHECK: DW_TAG_subroutine_type     DW_CHILDREN_yes
; CHECK-NEXT: DW_AT_rvalue_reference DW_FORM_flag_present
;
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG_subprogram
; CHECK:   DW_AT_name {{.*}}"l"
; CHECK-NOT: DW_TAG_subprogram
; CHECK:   DW_AT_reference [DW_FORM_flag_present] (true)

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG_subprogram
; CHECK:   DW_AT_name {{.*}}"r"
; CHECK-NOT: DW_TAG_subprogram
; CHECK:   DW_AT_rvalue_reference [DW_FORM_flag_present] (true)

%class.A = type { i8 }

; Function Attrs: nounwind
define void @_Z1gv() #0 {
  %a = alloca %class.A, align 1
  %pl = alloca { i64, i64 }, align 8
  %pr = alloca { i64, i64 }, align 8
  call void @llvm.dbg.declare(metadata !{%class.A* %a}, metadata !24, metadata !{metadata !"0x102"}), !dbg !25
  call void @llvm.dbg.declare(metadata !{{ i64, i64 }* %pl}, metadata !26, metadata !{metadata !"0x102"}), !dbg !31
  store { i64, i64 } { i64 ptrtoint (void (%class.A*)* @_ZNKR1A1lEv to i64), i64 0 }, { i64, i64 }* %pl, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata !{{ i64, i64 }* %pr}, metadata !32, metadata !{metadata !"0x102"}), !dbg !35
  store { i64, i64 } { i64 ptrtoint (void (%class.A*)* @_ZNKO1A1rEv to i64), i64 0 }, { i64, i64 }* %pr, align 8, !dbg !35
  ret void, !dbg !36
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @_ZNKR1A1lEv(%class.A*)

declare void @_ZNKO1A1rEv(%class.A*)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = metadata !{metadata !"0x11\004\00clang version 3.5 \000\00\000\00\000", metadata !1, metadata !2, metadata !3, metadata !16, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2\00A\002\008\008\000\000\000", metadata !5, null, null, metadata !6, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 2, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !"debug-info-qualifiers.cpp", metadata !""}
!6 = metadata !{metadata !7, metadata !13}
!7 = metadata !{metadata !"0x2e\00l\00l\00_ZNKR1A1lEv\005\000\000\000\006\0016640\000\005", metadata !5, metadata !"_ZTS1A", metadata !8, null, null, null, i32 0, metadata !12} ; [ DW_TAG_subprogram ] [line 5] [reference] [l]
!8 = metadata !{metadata !"0x15\00\000\000\000\000\0016384\000", i32 0, null, null, metadata !9, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [reference] [from ]
!9 = metadata !{null, metadata !10}
!10 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", null, null, metadata !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from ]
!11 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, null, metadata !"_ZTS1A"} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from _ZTS1A]
!12 = metadata !{i32 786468}
!13 = metadata !{metadata !"0x2e\00r\00r\00_ZNKO1A1rEv\007\000\000\000\006\0033024\000\007", metadata !5, metadata !"_ZTS1A", metadata !14, null, null, null, i32 0, metadata !15} ; [ DW_TAG_subprogram ] [line 7] [rvalue reference] [r]
!14 = metadata !{metadata !"0x15\00\000\000\000\000\0032768\000", i32 0, null, null, metadata !9, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [rvalue reference] [from ]
!15 = metadata !{i32 786468}
!16 = metadata !{metadata !17}
!17 = metadata !{metadata !"0x2e\00g\00g\00_Z1gv\0010\000\001\000\006\00256\000\0010", metadata !5, metadata !18, metadata !19, null, void ()* @_Z1gv, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 10] [def] [g]
!18 = metadata !{metadata !"0x29", metadata !5}         ; [ DW_TAG_file_type ]
!19 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !20, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!20 = metadata !{null}
!21 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!22 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!23 = metadata !{metadata !"clang version 3.5 "}
!24 = metadata !{metadata !"0x100\00a\0011\000", metadata !17, metadata !18, metadata !4} ; [ DW_TAG_auto_variable ] [a] [line 11]
!25 = metadata !{i32 11, i32 0, metadata !17, null}
!26 = metadata !{metadata !"0x100\00pl\0016\000", metadata !17, metadata !18, metadata !27} ; [ DW_TAG_auto_variable ] [pl] [line 16]
!27 = metadata !{metadata !"0x1f\00\000\000\000\000\000", null, null, metadata !28, metadata !"_ZTS1A"} ; [ DW_TAG_ptr_to_member_type ] [line 0, size 0, align 0, offset 0] [from ]
!28 = metadata !{metadata !"0x15\00\000\000\000\000\0016384\000", i32 0, null, null, metadata !29, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [reference] [from ]
!29 = metadata !{null, metadata !30}
!30 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", null, null, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!31 = metadata !{i32 16, i32 0, metadata !17, null}
!32 = metadata !{metadata !"0x100\00pr\0021\000", metadata !17, metadata !18, metadata !33} ; [ DW_TAG_auto_variable ] [pr] [line 21]
!33 = metadata !{metadata !"0x1f\00\000\000\000\000\000", null, null, metadata !34, metadata !"_ZTS1A"} ; [ DW_TAG_ptr_to_member_type ] [line 0, size 0, align 0, offset 0] [from ]
!34 = metadata !{metadata !"0x15\00\000\000\000\000\0032768\000", i32 0, null, null, metadata !29, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [rvalue reference] [from ]
!35 = metadata !{i32 21, i32 0, metadata !17, null}
!36 = metadata !{i32 22, i32 0, metadata !17, null}
