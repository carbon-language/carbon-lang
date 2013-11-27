; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=aranges - | FileCheck %s

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

%struct.foo = type { i8 }

@f = global %struct.foo zeroinitializer, align 1
@i = external global i32

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5 ", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !2, metadata !9, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/simple.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"simple.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"foo<&i>", i32 3, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, metadata !5, metadata !"_ZTS3fooIXadL_Z1iEEE"} ; [ DW_TAG_structure_type ] [foo<&i>] [line 3, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{i32 786480, null, metadata !"x", metadata !7, i32* @i, null, i32 0, i32 0} ; [ DW_TAG_template_value_parameter ]
!7 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !8} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !10}
!10 = metadata !{i32 786484, i32 0, null, metadata !"f", metadata !"f", metadata !"", metadata !11, i32 6, metadata !4, i32 0, i32 1, %struct.foo* @f, null} ; [ DW_TAG_variable ] [f] [line 6] [def]
!11 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/simple.cpp]
!12 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!13 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!14 = metadata !{metadata !"clang version 3.5 "}
