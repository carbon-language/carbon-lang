; struct A {
;   void foo();
; };
;  
; void (A::*p)() = &A::foo;
;
; RUN: llc -filetype=obj -o - %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; Check that the member function pointer is emitted without a DW_AT_size attribute.
; CHECK: DW_TAG_ptr_to_member_type
; CHECK-NOT: DW_AT_{{.*}}size
; CHECK: DW_TAG
;
; ModuleID = 'memberfnptr.cpp'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

%struct.A = type { i8 }

@p = global { i64, i64 } { i64 ptrtoint (void (%struct.A*)* @_ZN1A3fooEv to i64), i64 0 }, align 8

declare void @_ZN1A3fooEv(%struct.A*)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = !{!"0x11\004\00clang version 3.6.0 \000\00\000\00\001", !1, !2, !3, !2, !10, !2} ; [ DW_TAG_compile_unit ] [/memberfnptr.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"memberfnptr.cpp", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x13\00A\001\008\008\000\000\000", !1, null, null, !5, null, null, !"_ZTS1A"} ; [ DW_TAG_structure_type ] [A] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = !{!6}
!6 = !{!"0x2e\00foo\00foo\00_ZN1A3fooEv\002\000\000\000\000\00256\000\002", !1, !"_ZTS1A", !7, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 2] [foo]
!7 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null, !9}
!9 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!10 = !{!11}
!11 = !{!"0x34\00p\00p\00\005\000\001", null, !12, !13, { i64, i64 }* @p, null} ; [ DW_TAG_variable ] [p] [line 5] [def]
!12 = !{!"0x29", !1}                              ; [ DW_TAG_file_type ] [/memberfnptr.cpp]
!13 = !{!"0x1f\00\000\0064\000\000\000", null, null, !7, !"_ZTS1A"} ; [ DW_TAG_ptr_to_member_type ] [line 0, size 64, align 0, offset 0] [from ]
!14 = !{i32 2, !"Dwarf Version", i32 2}
!15 = !{i32 2, !"Debug Info Version", i32 2}
!16 = !{i32 1, !"PIC Level", i32 2}
!17 = !{!"clang version 3.6.0 "}
