; RUN: llc -filetype=asm -O0 -mtriple=x86_64-linux-gnu < %s | FileCheck %s
; RUN: llc -filetype=obj -O0 %s -mtriple=x86_64-linux-gnu -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s -check-prefix=CHECK-DWARF

; RUN: llc -filetype=obj %s -mtriple=x86_64-apple-darwin -o %t2
; RUN: llvm-dwarfdump %t2 | FileCheck %s -check-prefix=DARWIN-DWARF

; Testing case generated from:
; clang++ tu1.cpp tu2.cpp -g -emit-llvm -c
; llvm-link tu1.bc tu2.bc -o tu12.ll -S
; cat hdr.h
; struct foo {
; };
; cat tu1.cpp
; #include "hdr.h"
; foo f;
; cat tu2.cpp
; #include "hdr.h"
; foo g;

; Make sure we use relocation for ref_addr on non-darwin platforms.
; CHECK: DW_TAG_compile_unit
; CHECK: DW_TAG_variable
; CHECK: .long [[TYPE:.*]] # DW_AT_type
; CHECK: DW_TAG_structure_type
; CHECK: debug_info_begin1
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG_structure_type
; This variable's type is in the 1st CU.
; CHECK: DW_TAG_variable
; Make sure this is relocatable.
; CHECK: .quad .Lsection_info+[[TYPE]] # DW_AT_type
; CHECK-NOT: DW_TAG_structure_type
; CHECK: .section

; CHECK-DWARF: DW_TAG_compile_unit
; CHECK-DWARF: 0x[[ADDR:.*]]: DW_TAG_structure_type
; CHECK-DWARF: DW_TAG_compile_unit
; CHECK-DWARF: DW_TAG_variable
; CHECK-DWARF: DW_AT_type [DW_FORM_ref_addr] {{.*}}[[ADDR]])

; DARWIN-DWARF: DW_TAG_compile_unit
; DARWIN-DWARF: 0x[[ADDR:.*]]: DW_TAG_structure_type
; DARWIN-DWARF: DW_TAG_compile_unit
; DARWIN-DWARF: DW_TAG_variable
; DARWIN-DWARF: DW_AT_type [DW_FORM_ref_addr] {{.*}}[[ADDR]])

%struct.foo = type { i8 }

@f = global %struct.foo zeroinitializer, align 1
@g = global %struct.foo zeroinitializer, align 1

!llvm.dbg.cu = !{!0, !9}
!llvm.module.flags = !{!14, !15}

!0 = !{!"0x11\004\00clang version 3.4 (trunk 191799)\000\00\000\00\000", !1, !2, !3, !2, !6, !2} ; [ DW_TAG_compile_unit ] [/Users/manmanren/test-Nov/type_unique_air/ref_addr/tu1.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"tu1.cpp", !"/Users/manmanren/test-Nov/type_unique_air/ref_addr"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x13\00foo\001\008\008\000\000\000", !5, null, null, !2, null, null, !"_ZTS3foo"} ; [ DW_TAG_structure_type ] [foo] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = !{!"./hdr.h", !"/Users/manmanren/test-Nov/type_unique_air/ref_addr"}
!6 = !{!7}
!7 = !{!"0x34\00f\00f\00\002\000\001", null, !8, !4, %struct.foo* @f, null} ; [ DW_TAG_variable ] [f] [line 2] [def]
!8 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/Users/manmanren/test-Nov/type_unique_air/ref_addr/tu1.cpp]
!9 = !{!"0x11\004\00clang version 3.4 (trunk 191799)\000\00\000\00\000", !10, !2, !3, !2, !11, !2} ; [ DW_TAG_compile_unit ] [/Users/manmanren/test-Nov/type_unique_air/ref_addr/tu2.cpp] [DW_LANG_C_plus_plus]
!10 = !{!"tu2.cpp", !"/Users/manmanren/test-Nov/type_unique_air/ref_addr"}
!11 = !{!12}
!12 = !{!"0x34\00g\00g\00\002\000\001", null, !13, !4, %struct.foo* @g, null} ; [ DW_TAG_variable ] [g] [line 2] [def]
!13 = !{!"0x29", !10}        ; [ DW_TAG_file_type ] [/Users/manmanren/test-Nov/type_unique_air/ref_addr/tu2.cpp]
!14 = !{i32 2, !"Dwarf Version", i32 2}
!15 = !{i32 1, !"Debug Info Version", i32 2}
