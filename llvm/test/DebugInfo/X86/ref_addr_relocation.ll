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
; CHECK: debug_info_end0
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG_structure_type
; This variable's type is in the 1st CU.
; CHECK: DW_TAG_variable
; Make sure this is relocatable.
; CHECK: .quad .Lsection_info+[[TYPE]] # DW_AT_type
; CHECK-NOT: DW_TAG_structure_type
; CHECK: debug_info_end1

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

!0 = metadata !{metadata !"0x11\004\00clang version 3.4 (trunk 191799)\000\00\000\00\000", metadata !1, metadata !2, metadata !3, metadata !2, metadata !6, metadata !2} ; [ DW_TAG_compile_unit ] [/Users/manmanren/test-Nov/type_unique_air/ref_addr/tu1.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"tu1.cpp", metadata !"/Users/manmanren/test-Nov/type_unique_air/ref_addr"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x13\00foo\001\008\008\000\000\000", metadata !5, null, null, metadata !2, null, null, metadata !"_ZTS3foo"} ; [ DW_TAG_structure_type ] [foo] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !"./hdr.h", metadata !"/Users/manmanren/test-Nov/type_unique_air/ref_addr"}
!6 = metadata !{metadata !7}
!7 = metadata !{metadata !"0x34\00f\00f\00\002\000\001", null, metadata !8, metadata !4, %struct.foo* @f, null} ; [ DW_TAG_variable ] [f] [line 2] [def]
!8 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/Users/manmanren/test-Nov/type_unique_air/ref_addr/tu1.cpp]
!9 = metadata !{metadata !"0x11\004\00clang version 3.4 (trunk 191799)\000\00\000\00\000", metadata !10, metadata !2, metadata !3, metadata !2, metadata !11, metadata !2} ; [ DW_TAG_compile_unit ] [/Users/manmanren/test-Nov/type_unique_air/ref_addr/tu2.cpp] [DW_LANG_C_plus_plus]
!10 = metadata !{metadata !"tu2.cpp", metadata !"/Users/manmanren/test-Nov/type_unique_air/ref_addr"}
!11 = metadata !{metadata !12}
!12 = metadata !{metadata !"0x34\00g\00g\00\002\000\001", null, metadata !13, metadata !4, %struct.foo* @g, null} ; [ DW_TAG_variable ] [g] [line 2] [def]
!13 = metadata !{metadata !"0x29", metadata !10}        ; [ DW_TAG_file_type ] [/Users/manmanren/test-Nov/type_unique_air/ref_addr/tu2.cpp]
!14 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!15 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
