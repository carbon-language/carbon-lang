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
  call void @llvm.dbg.declare(metadata !{%class.A* %a}, metadata !24), !dbg !25
  call void @llvm.dbg.declare(metadata !{{ i64, i64 }* %pl}, metadata !26), !dbg !31
  store { i64, i64 } { i64 ptrtoint (void (%class.A*)* @_ZNKR1A1lEv to i64), i64 0 }, { i64, i64 }* %pl, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata !{{ i64, i64 }* %pr}, metadata !32), !dbg !35
  store { i64, i64 } { i64 ptrtoint (void (%class.A*)* @_ZNKO1A1rEv to i64), i64 0 }, { i64, i64 }* %pr, align 8, !dbg !35
  ret void, !dbg !36
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

declare void @_ZNKR1A1lEv(%class.A*)

declare void @_ZNKO1A1rEv(%class.A*)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5 ", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !16, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786434, metadata !5, null, metadata !"A", i32 2, i64 8, i64 8, i32 0, i32 0, null, metadata !6, i32 0, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 2, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !"debug-info-qualifiers.cpp", metadata !""}
!6 = metadata !{metadata !7, metadata !13}
!7 = metadata !{i32 786478, metadata !5, metadata !"_ZTS1A", metadata !"l", metadata !"l", metadata !"_ZNKR1A1lEv", i32 5, metadata !8, i1 false, i1 false, i32 0, i32 0, null, i32 16640, i1 false, null, null, i32 0, metadata !12, i32 5} ; [ DW_TAG_subprogram ] [line 5] [reference] [l]
!8 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 16384, null, metadata !9, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [reference] [from ]
!9 = metadata !{null, metadata !10}
!10 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from ]
!11 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !"_ZTS1A"} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from _ZTS1A]
!12 = metadata !{i32 786468}
!13 = metadata !{i32 786478, metadata !5, metadata !"_ZTS1A", metadata !"r", metadata !"r", metadata !"_ZNKO1A1rEv", i32 7, metadata !14, i1 false, i1 false, i32 0, i32 0, null, i32 33024, i1 false, null, null, i32 0, metadata !15, i32 7} ; [ DW_TAG_subprogram ] [line 7] [rvalue reference] [r]
!14 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 32768, null, metadata !9, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [rvalue reference] [from ]
!15 = metadata !{i32 786468}
!16 = metadata !{metadata !17}
!17 = metadata !{i32 786478, metadata !5, metadata !18, metadata !"g", metadata !"g", metadata !"_Z1gv", i32 10, metadata !19, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z1gv, null, null, metadata !2, i32 10} ; [ DW_TAG_subprogram ] [line 10] [def] [g]
!18 = metadata !{i32 786473, metadata !5}         ; [ DW_TAG_file_type ]
!19 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !20, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!20 = metadata !{null}
!21 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!22 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!23 = metadata !{metadata !"clang version 3.5 "}
!24 = metadata !{i32 786688, metadata !17, metadata !"a", metadata !18, i32 11, metadata !4, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 11]
!25 = metadata !{i32 11, i32 0, metadata !17, null}
!26 = metadata !{i32 786688, metadata !17, metadata !"pl", metadata !18, i32 16, metadata !27, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [pl] [line 16]
!27 = metadata !{i32 786463, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !28, metadata !"_ZTS1A"} ; [ DW_TAG_ptr_to_member_type ] [line 0, size 0, align 0, offset 0] [from ]
!28 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 16384, null, metadata !29, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [reference] [from ]
!29 = metadata !{null, metadata !30}
!30 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!31 = metadata !{i32 16, i32 0, metadata !17, null}
!32 = metadata !{i32 786688, metadata !17, metadata !"pr", metadata !18, i32 21, metadata !33, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [pr] [line 21]
!33 = metadata !{i32 786463, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !34, metadata !"_ZTS1A"} ; [ DW_TAG_ptr_to_member_type ] [line 0, size 0, align 0, offset 0] [from ]
!34 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 32768, null, metadata !29, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [rvalue reference] [from ]
!35 = metadata !{i32 21, i32 0, metadata !17, null}
!36 = metadata !{i32 22, i32 0, metadata !17, null}
