; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; From source:
; typedef void x;
; x *y;

; Check that a typedef with no DW_AT_type is produced. The absence of a type is used to imply the 'void' type.

; CHECK: DW_TAG_typedef
; CHECK-NOT: DW_AT_type
; CHECK: {{DW_TAG|NULL}}

@y = global i8* null, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = metadata !{metadata !"0x11\004\00clang version 3.5.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !2, metadata !3, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/typedef.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"typedef.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x34\00y\00y\00\002\000\001", null, metadata !5, metadata !6, i8** @y, null} ; [ DW_TAG_variable ] [y] [line 2] [def]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/typedef.cpp]
!6 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !7} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from x]
!7 = metadata !{metadata !"0x16\00x\001\000\000\000\000", metadata !1, null, null} ; [ DW_TAG_typedef ] [x] [line 1, size 0, align 0, offset 0] [from ]
!8 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!9 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!10 = metadata !{metadata !"clang version 3.5.0 "}

