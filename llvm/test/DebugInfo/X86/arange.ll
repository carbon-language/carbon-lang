; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj -generate-arange-section < %s | llvm-dwarfdump -debug-dump=aranges - | FileCheck %s
; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj -generate-arange-section < %s | llvm-readobj --relocations - | FileCheck --check-prefix=OBJ %s

; extern int i;
; template<int *x>
; struct foo {
; };
;
; foo<&i> f;

; Check that we only have one arange in this compilation unit (it will be for 'f'), and not an extra one (for 'i' - since it isn't actually defined in this CU)

; CHECK: Address Range Header
; CHECK-NEXT: [0x
; CHECK-NOT: [0x

; Check that we have a relocation back to the debug_info section from the debug_aranges section
; OBJ: debug_aranges
; OBJ-NEXT: R_X86_64_32 .debug_info 0x0

%struct.foo = type { i8 }

@f = global %struct.foo zeroinitializer, align 1
@i = external global i32

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = metadata !{metadata !"0x11\004\00clang version 3.5 \000\00\000\00\000", metadata !1, metadata !2, metadata !3, metadata !2, metadata !9, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/simple.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"simple.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x13\00foo<&i>\003\008\008\000\000\000", metadata !1, null, null, metadata !2, null, metadata !5, metadata !"_ZTS3fooIXadL_Z1iEEE"} ; [ DW_TAG_structure_type ] [foo<&i>] [line 3, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{metadata !"0x30\00x\000\000", null, metadata !7, i32* @i, null} ; [ DW_TAG_template_value_parameter ]
!7 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !8} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !10}
!10 = metadata !{metadata !"0x34\00f\00f\00\006\000\001", null, metadata !11, metadata !4, %struct.foo* @f, null} ; [ DW_TAG_variable ] [f] [line 6] [def]
!11 = metadata !{metadata !"0x29", metadata !1}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/simple.cpp]
!12 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!13 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!14 = metadata !{metadata !"clang version 3.5 "}
